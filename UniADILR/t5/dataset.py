

"""
Dataloading for EntailmentBank and RuleTaker.
"""
from copy import deepcopy
# from proof import Proof, InvalidProofStep
import random
import json
import itertools
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from utils import read_datas,extract_context,normalize
import re
from typing import *

import pytorch_lightning as pl

Example = Dict[str, Any]
Batch = Dict[str, Any]
# def read_proofs(path: str, is_train: bool) -> List[Example]:
#     """
#     Load the EntailmentBank dataset.
#     """
#     data = []
#     num_invalid = 0

#     for line in open(path):
#         ex = json.loads(line)
#         hypothesis = normalize(ex["hypothesis"])
#         #context = extract_context(ex["context"])
#         context = ex["context"]
#         proof_text = normalize(ex["proof"].strip())
#         print("proof_text",proof_text)
#         try:
#             proof = Proof(
#                 context,
#                 hypothesis,
#                 proof_text,
#                 strict=is_train,
#                 requires_complete=is_train,
#             )
#             data.append({"proof": proof})
#         except InvalidProofStep:
#             assert is_train
#             num_invalid += 1

#     print(f"{len(data)} proofs loaded. {num_invalid} invalid ones removed.")
#     return data


def serialize_context(context) -> str:
    return normalize(" ".join(f"{k}: {v}" for k, v in context.items()))

def shuffle_context(context,proof) :
    """
    Randomly shuffle the identifiers of the supporting facts.
    """
    num_sents = len(context)
    permutation = list(range(num_sents))
    random.shuffle(permutation)
    inv_permutation = [permutation.index(i) for i in range(num_sents)]

    shuffled_context = " ".join(
        f"sent{i+1}: {context[f'sent{permutation[i]+1}']}"
        for i in range(num_sents)
    )
    tokens = []
    premises = proof.split(" -> ")[0]
    hypothesis = proof.split(" -> ")[1]
    for t in premises.split(" & "):
        if re.fullmatch(r"sent\d+", t):
            i = int(t[4:])
            tokens.append(f"sent{inv_permutation[i-1]+1}")
        else:
            tokens.append(t)
    renamed_proof = " & ".join(tokens) + ' -> ' + hypothesis
    # print("----------------------------")
    # print(shuffled_context)
    # print(renamed_proof)
    # print("----------------------------")
    return shuffled_context,renamed_proof

class EntireProofDataset(Dataset):  # type: ignore
    def __init__(
        self,
        data: list,
        model_pth: str,
        max_input_len: int,
        max_output_len: int,
        pred_type: bool,
        is_train: bool,
        
    ) -> None:
        super().__init__()
        max_len = max(max_input_len, max_output_len)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_pth, model_max_length=max_len
        )
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.is_train = is_train
        self.pred_type = pred_type

        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = self.data[idx]
        hypothesis = normalize(ex["hypothesis"])
        context = ex["context"]
        proof = ex["proof"]
        type = ex["reasoning_type"]

        context_dict = context if isinstance(context,dict) else extract_context(context)

        if self.is_train:
            context,proof = shuffle_context(context_dict,proof)        #训练的时候，把context的顺序shuffle了一下（增加鲁棒性）
        
        context_text = serialize_context(context) if isinstance(context,dict) else context

        ex = deepcopy(ex)

        if not self.pred_type:
        
            input_seq = f"$hypothesis$ = {hypothesis} ; $context$ = {context_text}"
            ex["output_seq"] = proof
        else:
            input_seq = f"$hypothesis$ = {hypothesis} ; $context$ = {context_text}"
            ex["output_seq"] = proof + "; "+ f"$reasonint_type$ = {type}"

        ex["input_seq"] = input_seq
        
        return ex

    def collate(self, examples: List[Example]) -> Batch:
        inp = [ex["input_seq"] for ex in examples]
        input_seq = self.tokenizer(
            inp,
            padding="longest",
            max_length=self.max_input_len,
            truncation=True,
            return_tensors="pt",
        )

        oup = [ex["output_seq"] for ex in examples]
        output_seq = self.tokenizer(
            oup,
            padding="longest",
            max_length=self.max_output_len,
            truncation=True,
            return_tensors="pt",
        )
        output_seq.input_ids[output_seq.input_ids == self.tokenizer.pad_token_id] = -100

        batch = {
            "input_seq": inp,
            "input_seq_ids": input_seq.input_ids,
            "input_seq_mask": input_seq.attention_mask,
            "output_seq": oup,
            "output_seq_ids": output_seq.input_ids,
            "output_seq_mask": output_seq.attention_mask,
        }
        for k in examples[0].keys():
            if k not in ("input_seq", "output_seq"):
                batch[k] = [ex[k] for ex in examples]

        #print(batch)
        return batch



class ProofDataModels(pl.LightningDataModule):
    def __init__(
        self,
        model_path: str,
        max_input_len: int,
        max_output_len: int,
        batch_size: int,
        num_workers: int,
        train_path: list,
        val_path: list,
        test_path: list,
        pred_type: bool

    ) -> None:
        super().__init__()

        self.model_path = model_path
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.pred_type = pred_type

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:

        if stage in (None, "fit","validate"):

            train_datas = read_datas(self.train_path)
            val_datas = read_datas(self.val_path)


        if stage in (None, "fit"):

            self.ds_train = EntireProofDataset(  # type: ignore
                train_datas,
                self.model_path,
                self.max_input_len,
                self.max_output_len,
                self.pred_type,
                is_train=True
            )

        if stage in (None, "fit", "validate"):

            self.ds_val = EntireProofDataset(  # type: ignore
                val_datas,
                self.model_path,
                self.max_input_len,
                self.max_output_len,
                self.pred_type,
                is_train=False
            )

        if stage in (None, "test"):

            test_datas = read_datas(self.test_path)
            self.ds_test = EntireProofDataset(  # type: ignore
                test_datas,
                self.model_path,
                self.max_input_len,
                self.max_output_len,
                self.pred_type,
                is_train=False
            )

    def train_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.ds_train,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            pin_memory=False,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.ds_val,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            pin_memory=False,
            drop_last=True
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.ds_test,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.ds_test.collate,
            pin_memory=False,
            drop_last=False
        )


if __name__ == '__main__':
    print("000")

    train_path = ["/data/webw5/exp/BackChain/reasoning_type_exp/data/syn_data/deduction/deduction_train.jsonl",
                                    "/data/webw5/exp/BackChain/reasoning_type_exp/data/syn_data/induction/induction_test.jsonl"]


    train_datas = read_datas(train_path)
    train_set = EntireProofDataset(train_datas,
                                "/data/webw5/exp/BackChain/EntailmentBank/generate/t5-large",
                                512,512,True,True)
    
    print(len(train_set))
    
    
    
    train_loader =  DataLoader(
            train_set,
            4,
            shuffle = True,
            num_workers = 0,
            collate_fn = train_set.collate,
            pin_memory = False,
            drop_last = True
        )
    

    for i,data in enumerate(train_loader):
        print(data)
        if i > 10:
            break

    print(len(train_loader))