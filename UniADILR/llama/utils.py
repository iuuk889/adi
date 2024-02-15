


import json
import random


from typing import *
from collections import OrderedDict

import re

import unicodedata





def read_datas(path_ls):
    datas = []

    for data_path in path_ls:
        for line in open(data_path):
            ex = json.loads(line)
            datas.append(ex)
    
    random.shuffle(datas)
    random.shuffle(datas)
    return datas  #返回一个包含所有数据的data列表，里面每个元素是一个list



def serialize_context(context) -> str:
    return normalize("\n".join(f"{k}: {v}" for k, v in context.items()))


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


def jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    similarity = intersection / union if union != 0 else 0.0
    
    return similarity


def evaluate(results,pred_typing,bleurt_scorer = None,bleurt_tokenizer = None):

    proof_crr = 0
    conclu_crr = 0
    type_crr = 0
    proof_one_acc = 0

    for result in results:
        #print(result)
        pred_proof = result['proof_pred']
        gt_proof = result['proof_gt']

        pred_conclusion = result['conclusion_pred']
        gt_conclusion = result['conclusion_label']

        if set(pred_proof) == set(gt_proof):
            proof_crr += 1

        else:
            for p in pred_proof:
                if p in gt_proof:
                    proof_one_acc += 1
        
        # sim_score = bleurt_scorer(**bleurt_tokenizer(pred_conclusion,gt_conclusion,return_tensors='pt',padding=True,truncation=True))
        # sim_score = sim_score.logits

        sim_score = jaccard_similarity(pred_conclusion,gt_conclusion)
        if sim_score > 0.9:
            conclu_crr += 1

        if pred_typing and result['type_pred'] == result['type_label']:
            type_crr += 1

    if pred_typing:
        return {"proof_acc":proof_crr /len(results),
                "proof_rank_one_acc":proof_one_acc/len(results),
                "conclusion_acc":conclu_crr / len(results),
                "type_acc":type_crr / len(results)}
    else:
        return {"proof_acc":proof_crr /len(results),
                "conclusion_acc":conclu_crr / len(results),
}
    


    

def parse_proof(item):

    proof = item['proof']
    premises = proof.split(" -> ")[0]
    proof_ls = premises.split(" & ")
    conclusion = normalize_sentence(item['hypothesis'])
    type_label = item["reasoning_type"]


    return proof_ls,conclusion,type_label