
from typing import List, Optional
# import fire
# from llama import Llama, Dialog
# import torch.distributed as dist
import os
import json
import random
from utils import *
import time
import re


import os

import openai
#openai.api_key = "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"
# os.environ["http_proxy"] = 'http://localhost:7890'
# os.environ["https_proxy"] = 'http://localhost:7890'
# os.environ["http_proxy"] = "http://localhost:7890"
# os.environ["https_proxy"] = "http://localhost:7890"s
from openai import OpenAI
client = OpenAI(api_key= "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")

all_prompt = 'Select the logical supporting facts from the context to prove the hypothesis and output a proof. \
Additionally, output the type of logical relationship employed in proving the hypothesis, \
choosing from abduction, deduction, and induction.\nSome Examples are shown:\n\
Example1.:\n\
"hypothesis": {a_hypothesis}\n"context": {a_context_text}\nOutput:\nProof = {a_proof}\nReasoning Type = {a_reasoning_type}\n\
Example2.:\n\
"hypothesis": {d_hypothesis}\n"context": {d_context_text}\nOutput:\nProof = {d_proof}\nReasoning type = {d_reasoning_type}\n\
Example3.:\n \
"hypothesis": {i_hypothesis}\n"context": {i_context_text}\nOutput:\nProof = {i_proof}\nReasoning type = {i_reasoning_type}\n\
Now answer the question:\n "hypothesis": {test_hypothesis} \n "context": {test_context_text}'


def extract_answer(text):

    proof_match = re.search(r'Proof = (.*? -> .*?)\n', text, re.DOTALL)
    reasoning_type_match = re.search(r'Reasoning Type = (.*?)\n', text)


    # 输出匹配结果
    if proof_match:
        proof = proof_match.group(1).strip()
        proof = proof.replace("[","")
        proof = proof.replace("]","")
        proof_list = re.findall(r'sent\d+', proof)
        if " -> " in proof:
            conclusion = proof.split(" -> ")[1]
            start_index = conclusion.find(" Reasoning Type = ")

            # 如果找到了该子字符串
            if start_index != -1:
                # 截取字符串，去掉该子字符串及其之后的部分
                conclusion = conclusion[:start_index]

        else:
            conclusion = 'none'
    else:
        proof_list = []
        conclusion = 'none'

    if reasoning_type_match:
        reasoning_type = reasoning_type_match.group(1)
    else:
        reasoning_type = 'none'

    return proof_list,conclusion,reasoning_type





# def main(
#     ckpt_dir: str = '/data/webw5/exp/BackChain/Llama-2-13b-chat',
#     tokenizer_path: str = '/data/webw5/exp/BackChain/Llama-2-13b-chat/tokenizer.model',
#     temperature: float = 0.9,
#     top_p: float = 0.9,
#     max_seq_len: int = 1024,  #1024  最大输入长度限制，不能输入过长
#     max_batch_size: int = 8,
#     max_gen_len: Optional[int] = 200,
#     with_descript: bool = False,
#     prompt_reasoning_type:str = 'induction'

# ):

    # zero-shot的模板不需要示例，只需要test的hypothesis 和context

abduction_prompt_set = read_datas(["./data/abduction_train.jsonl"])
deduction_prompt_set = read_datas(["./data/deduction_train.jsonl"])
induction_prompt_set = read_datas(["./data/induction_train.jsonl"])

prompt_template = all_prompt

#现在只做了单一提示和测试
test_paths = ['./data/abduction_test.jsonl',
            #'/data/webw5/exp/BackChain/reasoning_type_exp/data/syn_data/induction/induction_test.jsonl',
            './data/deduction_test.jsonl',
            # '/data/webw5/exp/BackChain/reasoning_type_exp/data/human_data/abduction.jsonl',
            # '/data/webw5/exp/BackChain/reasoning_type_exp/data/human_data/deduction.jsonl',
            # '/data/webw5/exp/BackChain/reasoning_type_exp/data/human_data/induction.jsonl'
            ]
#test_path = ['/data/webw5/exp/BackChain/reasoning_type_exp/data/syn_data/abduction/abduction_test.jsonl']

metrics = []

for i,test_path in enumerate(test_paths):
    save_path = f'./results/gpt_35_all_{i}.jsonl'
    
    test_datas = read_datas([test_path])
    choiced_test_datas = random.sample(test_datas,2)

    results = []

    start_time = time.time()

    for ex in choiced_test_datas:

        a_prompt_example = random.choice(abduction_prompt_set)
        d_prompt_example = random.choice(deduction_prompt_set)
        i_prompt_example = random.choice(induction_prompt_set)

        hypothesis = normalize_sentence(ex["hypothesis"])
        context = ex["context"]
        context_text = serialize_context(context) if isinstance(context,dict) else context
        a_prompt_context_text = serialize_context(a_prompt_example['context']) if isinstance(a_prompt_example['context'],dict) else a_prompt_example['context']
        d_prompt_context_text = serialize_context(d_prompt_example['context']) if isinstance(d_prompt_example['context'],dict) else d_prompt_example['context']
        i_prompt_context_text = serialize_context(i_prompt_example['context']) if isinstance(i_prompt_example['context'],dict) else i_prompt_example['context']            
        content = prompt_template.format(
            a_hypothesis = normalize_sentence(a_prompt_example['hypothesis']),
            a_context_text = a_prompt_context_text,
            a_proof = a_prompt_example['proof'],
            a_reasoning_type = a_prompt_example['reasoning_type'],
            d_hypothesis = normalize_sentence(d_prompt_example['hypothesis']),
            d_context_text = d_prompt_context_text,
            d_proof = d_prompt_example['proof'],
            d_reasoning_type = d_prompt_example['reasoning_type'],
            i_hypothesis = normalize_sentence(i_prompt_example['hypothesis']),
            i_context_text = i_prompt_context_text,
            i_proof = i_prompt_example['proof'],
            i_reasoning_type = i_prompt_example['reasoning_type'],
            test_hypothesis = hypothesis,
            test_context_text = context_text)
    

        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content":content}
        ]
        )

        response = completion.choices[0].message
        
        #response_text = ''.join(c for c in response)
        print(response)

        print("======================================")

        gold_proof_ls,gold_conclusion,gold_type_label = parse_proof(ex)
        # print("Response\n",response)
        pred_proofs,pred_conclusion,pred_type = extract_answer(response)
        print("Proofs",pred_proofs)
        print("Conclusion",pred_conclusion)
        print("Pred_reasoning_type",pred_type)
        print("======================================")

        num = 0
        # while len(pred_proofs) == 0 and num < 5:

        #     response = get_response(content = revise_prompt)
        #     print(response)

        #     pred_proofs,pred_conclusion,pred_type = extract_answer(response)
        #     num += 1

        item = {"original_ouput":response,
                        
                        "proof_gt":gold_proof_ls,
                        "conclusion_label":gold_conclusion,
                        "type_label":gold_type_label,
                        
                        "proof_pred":pred_proofs,
                        "conclusion_pred":pred_conclusion,
                        "type_pred":pred_type
                        }

        results.append(item)

        with open(save_path, 'a+') as jsonl_file:

            json_line = json.dumps(item) + '\n'  # 将每个对象转换为 JSON 格式并加上换行符
            jsonl_file.write(json_line)


    print("Consumed time", str(time.time() - start_time))

    metric = evaluate(results,pred_typing = True)
    for k,v in metric.items():
        print(f'{k}:{v}')
    metrics.append(metric)

for path,metric in zip(test_paths,metrics):
    print("------------------------------------")
    print(path)
    for k,v in metric.items():
        print(f'{k}:{v}')
    print("------------------------------------")
        
# if __name__ == "__main__":
#     fire.Fire(main)