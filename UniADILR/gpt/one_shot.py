


#测一下one-shot，比较在（a,d)和-shot和all-shot的区别
from typing import List, Optional
# from llama import Llama, Dialog
# import torch.distributed as dist
import os
import json
import random
from utils import *
import time
import re
import openai

openai.api_key = "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"

prompt_reasoning_type = 'deduction'

single_shot_prompt = 'Select the logical supporting facts from the context to prove the hypothesis and output a proof. \
Additionally, output the type of logical relationship employed in proving the hypothesis, \
choosing from abduction, deduction, and induction.\nExample.\n"hypothesis": {example_hypothesis}\n\
"context": {example_context_text}\nOutput:\n\
Proof = {example_proof} \nReasoning Type = {example_reasoning_type}\n\
Now answer the question:\n "hypothesis": {test_hypothesis} \n "context": {test_context_text}'

def extract_answer(text):

    proof_match = re.search(r'Proof = (.*? -> .*?)\n', text, re.DOTALL)
    reasoning_type_match = re.search(r'Reasoning Type = (\w+)', text)


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



#现在只做了单一提示和测试
test_paths = ['./data/abduction_test.jsonl',
            #'./data/deduction_test.jsonl',
            #'./data/induction_test.jsonl',
            # '/data/webw5/exp/BackChain/reasoning_type_exp/data/human_data/abduction.jsonl',
            # '/data/webw5/exp/BackChain/reasoning_type_exp/data/human_data/deduction.jsonl',
            # '/data/webw5/exp/BackChain/reasoning_type_exp/data/human_data/induction.jsonl'
            ]
#test_path = ['/data/webw5/exp/BackChain/reasoning_type_exp/data/syn_data/abduction/abduction_test.jsonl']

metrics = []


if prompt_reasoning_type == 'abduction':
    prompt_set = read_datas(["./data/abduction_train.jsonl"])

elif prompt_reasoning_type == 'deduction':
    prompt_set = read_datas(["./data/deduction_train.jsonl"])

elif prompt_reasoning_type == 'induction':
    prompt_set = read_datas(["./data/induction_train.jsonl"])


for i,test_path in enumerate(test_paths):
    save_path = f'./results/gpt_4_one_shot_{prompt_reasoning_type}_{i}.jsonl'
    
    test_datas = read_datas([test_path])
    choiced_test_datas = random.sample(test_datas,100)

    results = []

    start_time = time.time()

    for ex in choiced_test_datas:

        prompt_example = random.choice(prompt_set)
        hypothesis = normalize_sentence(ex["hypothesis"])
        context = ex["context"]
        context_text = serialize_context(context) if isinstance(context,dict) else context
        
        prompt_context_text = serialize_context(prompt_example['context']) if isinstance(prompt_example['context'],dict) else prompt_example['context']

        content = single_shot_prompt.format(
            example_hypothesis = normalize_sentence(prompt_example['hypothesis']),
            example_context_text =  prompt_context_text,
            example_proof = prompt_example['proof'],
            example_reasoning_type = prompt_example['reasoning_type'],
            test_hypothesis = hypothesis,
            test_context_text = context_text)

        gold_proof_ls,gold_conclusion,gold_type_label = parse_proof(ex)
        # print(gold_proof_ls)
        # print(gold_conclusion)
        # print(gold_type_label)
        # print(content)
    
        completion = openai.ChatCompletion.create(
                model="gpt-4",  
                messages=[
                    {"role": "user", "content": content},
                        ]
        )

        response = completion.choices[0].message
        
        response_text = response["content"]   #''.join(c for c in response)

        pred_proofs,pred_conclusion,pred_type = extract_answer(response_text)
        print("--------------------------------")
        print(response_text)
        print("Gold proofs",gold_proof_ls)
        print("Pred Proofs",pred_proofs)
        print("Gold_reasoning_type",gold_type_label)
        print("Pred_reasoning_type",pred_type)
        print("--------------------------------")

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
        
