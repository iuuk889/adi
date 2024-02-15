
# from common import *
# from proof import ProofStep, Proof, InvalidProofStep
# from search import ProofGraph
import numpy as np
import os
import json
import torch
import itertools
import pytorch_lightning as pl
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    LogitsProcessor,
    AutoModelForSequenceClassification
)

from utils import evaluate,get_optimizers

from typing import *

bleurt_path = '/data/webw5/exp/BackChain/EntailmentBank/generate/bleurt-tiny-512'
bleurt_tokenizer = AutoTokenizer.from_pretrained(bleurt_path)
bleurt_scorer = AutoModelForSequenceClassification.from_pretrained(bleurt_path)



Example = Dict[str, Any]
Batch = Dict[str, Any]
# Some handcrafted heuristics for constraining the predicted proof steps.
# They often make the proof graph less cluttered but do not improve the final performance.
# So we do not use them by default.
# class PermutationInvarianceLogitsProcessor(LogitsProcessor):
#     def __init__(
#         self, num_beams: int, context: List[OrderedDict[str, str]], tokenizer: Any
#     ) -> None:
#         self.num_beams = num_beams
#         self.context = context
#         self.tokenizer = tokenizer
#         self.semicolon_token_id = tokenizer.convert_tokens_to_ids(";")

#     def __call__(
#         self, input_ids: torch.LongTensor, scores: torch.FloatTensor
#     ) -> torch.FloatTensor:
#         generated_texts = self.tokenizer.batch_decode(
#             input_ids, skip_special_tokens=True
#         )

#         batch_size = input_ids.size(0) // self.num_beams
#         unique_premises: List[Set[Any]] = [set() for _ in range(batch_size)]

#         for i, prefix in enumerate(generated_texts):
#             if "->" in prefix:  # conclusion
#                 if prefix.count("->") > 1:
#                     scores[i, :] = float("-inf")
#                     continue
#                 concl = prefix.split("->")[1].strip()
#                 if concl == "hypothesis":
#                     # Only ";" after "-> hypothesis".
#                     s = scores[i, self.semicolon_token_id].item()
#                     scores[i, :] = float("-inf")
#                     scores[i, self.semicolon_token_id] = s
#                 elif ";" in concl:
#                     # Must end after ";"
#                     s = scores[i, self.tokenizer.eos_token_id].item()
#                     scores[i, :] = float("-inf")
#                     scores[i, self.tokenizer.eos_token_id] = s
#                 elif (
#                     concl != ""
#                     and not concl.startswith("int")
#                     and not "int".startswith(concl)
#                 ):
#                     # The conclusion is either the hypothesis or an intermediate.
#                     scores[i, :] = float("-inf")
#                 elif "-> int" in prefix:
#                     # Only one conclusion for fixed premises.
#                     j = scores[i, :].argmax()
#                     s = scores[i, j].item()
#                     scores[i, :] = float("-inf")
#                     scores[i, j] = s

#             else:  # premises
#                 n = i // self.num_beams
#                 premises = tuple(sorted([p.strip() for p in prefix.split("&")]))
#                 if premises in unique_premises[n] or len(set(premises)) < len(premises):
#                     scores[i, :] = float("-inf")
#                     continue
#                 unique_premises[n].add(premises)

#                 tokens = prefix.split()
#                 for t in tokens[:-1]:
#                     if t != "&" and re.fullmatch(r"(int|sent)\d+", t) == None:
#                         scores[i, :] = float("-inf")
#                     elif (
#                         re.fullmatch(r"sent\d+", t) != None and t not in self.context[n]
#                     ):
#                         scores[i, :] = float("-inf")
#                 if len(tokens) >= 1:
#                     t = tokens[-1]
#                     if (
#                         t != "&"
#                         and re.fullmatch(r"(int|sent)\d+", t) == None
#                         and not "sent".startswith(t)
#                         and not "int".startswith(t)
#                     ):
#                         scores[i, :] = float("-inf")

#         return scores


class ProofModel(pl.LightningModule):
    def __init__(
        self,
        model_path: str,
        lr: float,
        warmup_steps: int,
        num_beams: int,
        topk: int,
        max_input_len: int,
        pred_type:bool
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_beams = num_beams
        self.topk = topk

        self.pred_type = pred_type

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, model_max_length=max_input_len
        )
        if "t5" in model_path:
            self.seq2seq = T5ForConditionalGeneration.from_pretrained(model_path)
        elif "bart" in model_path:
            self.seq2seq = BartForConditionalGeneration.from_pretrained(model_path)
        else:
            raise NotImplementedError

    def forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Any:
        return self.seq2seq(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss


    def on_train_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)  # type: ignore
            assert self.trainer is not None
            print(f"Logging to {self.trainer.log_dir}")


    def generate_entire_proof(
        self, input_text: List[str]
    ) -> Tuple[List[str], List[float]]:
        """
        Single-shot proof generation with text-to-text transformers.
        """
        assert self.trainer is not None

        # print("----------------------------------")
        # print(input_text)
        # print("----------------------------------")
        input = self.tokenizer(
            input_text,
            padding="longest",
            max_length=self.trainer.datamodule.max_input_len,  # type: ignore
            truncation=True,
            return_tensors="pt",
        )
        output = self.seq2seq.generate(
            input_ids=input.input_ids.to(self.device, non_blocking=True),
            attention_mask=input.attention_mask.to(self.device, non_blocking=True),
            max_length=self.trainer.datamodule.max_output_len,  # type: ignore
            num_beams=self.num_beams,
            num_return_sequences=1,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )  #只返回了一个

        output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        scores = output.sequences_scores.detach().exp().tolist()

        #print(output_text)
        return output_text, scores

    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
        
        # print("8888888888888888")
        # print(batch['input_seq'])
        # print(batch['output_seq'])

        # print("8888888888888888")

        loss = self(
            batch["input_seq_ids"],
            batch["input_seq_mask"],
            batch["output_seq_ids"],
        )
        self.log("loss_train", loss, on_epoch=True, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch: Batch, batch_idx: int) -> Tuple[Any]:  # type: ignore
        return self.val_test_step("val", batch, batch_idx)

    def test_step(self, batch: Batch, batch_idx: int) -> Tuple[Any]:  # type: ignore
        return self.val_test_step("test", batch, batch_idx)

    def validation_epoch_end(self, outputs: Iterable[Any]) -> None:
        return self.val_test_epoch_end("val", outputs)

    def test_epoch_end(self, outputs: Iterable[Any]) -> None:
        return self.val_test_epoch_end("test", outputs)

    def val_test_step(self, split: str, batch: Batch, batch_idx: int) -> Tuple[Any]:

        loss = self(
            batch["input_seq_ids"],
            batch["input_seq_mask"],
            batch["output_seq_ids"],
        )
        self.log(f"loss_{split}", loss, sync_dist=True)
        proof_pred, score = self.generate_entire_proof(batch["input_seq"])

        #处理一个batch和test的模块
        #print("batch",batch)

        batched_data =  [{key: value 
                        for key, value in zip(batch.keys(), values)} 
                        for values in zip(*batch.values())] #处理成一个列表，每个列表里面是一个字典
        
        #print("batched_data",batched_data)

        return proof_pred, score, batched_data


    def parse_proof(self,proof_text,item,pred_or_gt):

        if pred_or_gt == 'pred':
            if self.pred_type:

                if " ; " not in proof_text:
                    return ["invalid"],'invalid','invalid'

                proof_pred = proof_text.split("; ")[0]

                if " -> " not in proof_pred:
                    return ["invalid"],'invalid','invalid'

                premises = proof_pred.split(" -> ")[0]
                proof_ls = premises.split(" & ")
                if " -> " in proof_pred:
                    conclusion = proof_pred.split(" -> ")[1]
                    conclusion = conclusion.strip(". ")
                else:
                    conclusion = "none"
                
                type_label = proof_text.split("; ")[1]
                type_label = type_label.strip("$reasonint_type$ = ",'')

            else:
                if " -> " not in proof_text:
                    return ["invalid"],'invalid','invalid'
                else:
                    premises = proof_text.split(" -> ")[0]
                    proof_ls = premises.split(" & ")

                    conclusion = proof_text.split(" -> ")[1]
                    type_label = 'none'

        elif pred_or_gt == 'gold':
            proof = item['proof']
            premises = proof.split(" -> ")[0]
            proof_ls = premises.split(" & ")
            conclusion = item['hypothesis']
            type_label = item["reasoning_type"]

        else:
            raise NotImplementedError

        return proof_ls,conclusion,type_label

    def val_test_epoch_end(self, split: str, outputs: Iterable[Any]) -> None:
        results = []

        for out in outputs:
            #print("out",out)

            for proof_pred, score, item in zip(*out):
                # print("--------------------")
                # print(item)
                # print("--------------------")

                proof_pred,conclusion_pred,type_pred = self.parse_proof(proof_pred,item,"pred")
                proof_label,conclusion_label,type_label = self.parse_proof(proof_pred,item,"gold")

                results.append(
                    {
                        "hypothesis": item["hypothesis"],
                        "context": item["context"],

                        "proof_gt": proof_label,
                        "conclusion_label":conclusion_label,
                        "type_label": type_label,

                        "proof_pred": proof_pred,
                        "conclusion_pred": conclusion_pred,
                        "type_pred": type_pred
                    }
                )


        assert self.trainer is not None
        if self.logger is not None and self.trainer.log_dir is not None:
            json_path = os.path.join(self.trainer.log_dir, f"results_{split}.json")
            json.dump(results, open(json_path, "a"))
            print(f"Validation results saved to {json_path}")  #将valid的结果写入到文件中

        metric = evaluate(results,self.pred_type,bleurt_scorer,bleurt_tokenizer)

        for k, v in metric.items():
            self.log(f"{k}_{split}", v, on_step=False, on_epoch=True)
            print(f"{k}_{split}:",v)


    def configure_optimizers(self) -> Dict[str, Any]:
        assert self.trainer is not None
        if self.trainer.max_steps != -1:
            max_steps = self.trainer.max_steps
        else:
            max_steps = (
                self.trainer.max_epochs
                * len(self.trainer.datamodule.train_dataloader())  # type: ignore
                // self.trainer.accumulate_grad_batches
            )
        return get_optimizers(
            self.parameters(),
            self.lr,
            self.warmup_steps,
            max_steps,
        )
