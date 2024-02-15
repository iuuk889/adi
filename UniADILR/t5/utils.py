

import json
import random


from typing import *
from collections import OrderedDict

from fairseq.optim.adafactor import FairseqAdafactor,Adafactor
# from fairseq.optim import adafactor

import torch_optimizer
# from fairscale import Adamfac
def read_datas(path_ls):
    datas = []

    for data_path in path_ls:
        for line in open(data_path):
            ex = json.loads(line)
            datas.append(ex)
    
    random.shuffle(datas)
    random.shuffle(datas)
    return datas  #返回一个包含所有数据的data列表，里面每个元素是一个list




import re
import torch
import random
import unicodedata

from transformers import get_cosine_schedule_with_warmup
from typing import *

Example = Dict[str, Any]
Batch = Dict[str, Any]


def normalize(text: str) -> str:
    """
    Deal with unicode-related artifacts.
    """
    return unicodedata.normalize("NFD", text)


def normalize_sentence(text: str) -> str:
    """
    Convert sentences to lowercase and remove the trailing period.
    """
    text = normalize(text).lower().strip()
    if text.endswith("."):
        text = text[:-1].strip()
    return text


def extract_context(ctx: str) : #-> OrderedDict[str, str]
    """
    Extract supporting facts from string to dict.
    """
    return OrderedDict(
        {
            ident.strip(): normalize_sentence(sent)
            for ident, sent in re.findall(
                r"(?P<ident>sent\d+): (?P<sent>.+?) (?=sent)", ctx + " sent"
            )
        }
    )



    

def evaluate(results,pred_typing,bleurt_scorer,bleurt_tokenizer):

    proof_crr = 0
    conclu_crr = 0
    type_crr = 0

    for result in results:
        pred_proof = result['proof_pred']
        gt_proof = result['proof_gt']

        pred_conclusion = result['conclusion_pred']
        gt_conclusion = result['conclusion_label']

        if set(pred_proof) == set(gt_proof):
            proof_crr += 1
        
        sim_score = bleurt_scorer(**bleurt_tokenizer(pred_conclusion,gt_conclusion,return_tensors='pt',padding=True,truncation=True))
        sim_score = sim_score.logits

        
        if sim_score > 0.75:
            conclu_crr += 1

        if pred_typing and result['type_pred'] == result['type_label']:
            type_crr += 1

    if pred_typing:
        return {"proof_acc":proof_crr /len(results),
                "conclusion_acc":conclu_crr / len(results),
                "type_acc":type_crr / len(results)}
    else:
        return {"proof_acc":proof_crr /len(results),
                "conclusion_acc":conclu_crr / len(results),
}
        
def get_optimizers(
    parameters: Iterable[torch.nn.parameter.Parameter],
    lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
) -> Dict[str, Any]:
    """
    Get an AdamW optimizer with linear learning rate warmup and cosine decay.
    """
    #print(lr)
    #optimizer = torch.optim.AdamW(parameters, lr=lr)
    optimizer = Adafactor(parameters, lr=lr,relative_step = False)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        },
    }
