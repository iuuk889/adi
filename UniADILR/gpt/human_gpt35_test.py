from typing import List, Optional
# from llama import Llama, Dialog
# import torch.distributed as dist
import os
import json
import random
from utils import *
import time
import re
import os
# os.environ["REPLICATE_API_TOKEN"] ='r8_BlHB5FAlfubOrGMzg7hwfT6JzbOrg6Z0TKQiX'
# import replicate

import openai
openai.api_key = "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"

zero_shot_prompt = 'Please select the logical supporting facts from the context to prove the hypothesis and output a proof. Additionally, output the type of logical relationship employed in proving the hypothesis, choosing from abduction, deduction, and induction. \n\
The output should be in the following format and don\'t output any additional text otherwise:\n\
$Output Example$:\n Proof = sent[number1] & sent[number2] -> {test_hypothesis}\nReasoning Type = abduction[choose one from abduction, deduction, and induction]\n\
$Now answer the question$:\n $To prove hypothesis$: {test_hypothesis} \n $context$: {test_context_text}\n'
        

single_shot_prompt = 'Select the logical supporting facts from the context to prove the hypothesis and output a proof. \
Additionally, output the type of logical relationship employed in proving the hypothesis, \
choosing from abduction, deduction, and induction.\nExample.\n"hypothesis": {example_hypothesis}\n\
"context": {example_context_text}\nOutput:\n\
Proof = {example_proof} \nReasoning Type = {example_reasoning_type}\n\
Now answer the question:\n "hypothesis": {test_hypothesis} \n "context": {test_context_text}'


def extract_answer(text):


    #proof_match = re.search(r'Proof = (.*? -> .*?)\n', text, re.DOTALL)

    proof_match = re.findall(r'sent\d+', text)
    reasoning_type_match = re.search(r'Reasoning Type = (\w+)', text)  #.*?


    # 输出匹配结果
    if proof_match:
        proof = proof_match#.group(1).strip()
        # proof = proof.replace("[","")
        # proof = proof.replace("]","")
        # proof_list = re.findall(r'sent\d+', proof)
        proof_list = proof

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



test_type= 'deduction'

if test_type == 'abduction':
    test_path = ['./data/human_data/abduction.jsonl']
    id_dataset = read_datas(['./data/abduction_train.jsonl'])
    ood_dataset = read_datas(['./data/deduction_train.jsonl',
                            './data/induction_train.jsonl' ])
    random.shuffle(id_dataset)
    random.shuffle(ood_dataset)

elif test_type == 'deduction':

    test_path = ['./data/human_data/deduction_new_complete.jsonl']
    id_dataset = read_datas(['./data/deduction_train.jsonl'])
    ood_dataset = read_datas(['./data/induction_train.jsonl',
                            './data/abduction_train.jsonl'])
    random.shuffle(id_dataset)
    random.shuffle(ood_dataset)
    
elif test_type == 'induction':

    test_path = ['./data/human_data/induction.jsonl']
    id_dataset = read_datas(['./data/induction_train.jsonl'])
    ood_dataset = read_datas(['./data/abduction_train.jsonl',
                            './data/deduction_train.jsonl'])
    random.shuffle(id_dataset)
    random.shuffle(ood_dataset)     

else:
    raise ValueError

test_datas = read_datas(test_path)
choiced_test_datas = random.sample(test_datas,200)

result_dict = {"0-shot":{},"id-shot":{},"ood-random-shot":{}}

# 0-shot

results = []
save_path = f'./results/human_results/new_gpt_35_human_{test_type}_0_shot.jsonl'

for ex in choiced_test_datas:

    hypothesis = normalize_sentence(ex["hypothesis"])
    context = ex["context"]
    context_text = serialize_context(context) if isinstance(context,dict) else context

    gold_proof_ls,gold_conclusion,gold_type_label = parse_proof(ex)
    
    content = zero_shot_prompt.format(test_hypothesis = hypothesis,
                                    test_context_text = context_text)
    
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "user", "content": content},
                ])

    response = completion.choices[0].message

    response_text = response["content"]   #''.join(c for c in response)
    print(response_text)


    pred_proofs,pred_conclusion,pred_type = extract_answer(response_text)
    print("--------------------------------")
    print("Gold proofs",gold_proof_ls)
    print("Pred Proofs",pred_proofs)
    print("Gold_reasoning_type",gold_type_label)
    print("Pred_reasoning_type",pred_type)
    print("--------------------------------")


    item = {"original_ouput":response,
                        
                        "proof_gt":gold_proof_ls,
                        "conclusion_label":gold_conclusion,
                        "type_label":gold_type_label,
                        
                        "proof_pred":pred_proofs,
                        "conclusion_pred":pred_conclusion,
                        "type_pred":pred_type
                        }
    
    print("--------------------------------")

    results.append(item)

    with open(save_path, 'a+') as jsonl_file:
        json_line = json.dumps(item) + '\n'  # 将每个对象转换为 JSON 格式并加上换行符
        jsonl_file.write(json_line)

metric = evaluate(results,pred_typing = True)
result_dict['0-shot'] = metric




# ID-shot

results = []
save_path = f'./results/human_results/new_gpt_35_human_{test_type}_1_shot_id.jsonl'

for ex in choiced_test_datas:

    prompt_example = random.choice(id_dataset)

    hypothesis = normalize_sentence(ex["hypothesis"])
    context = ex["context"]
    context_text = serialize_context(context) if isinstance(context,dict) else context


    example_hypothesis = normalize_sentence(prompt_example["hypothesis"])
    example_context = prompt_example["context"]
    example_context_text = serialize_context(example_context) if isinstance(example_context,dict) else example_context


    gold_proof_ls,gold_conclusion,gold_type_label = parse_proof(ex)

    
    content = single_shot_prompt.format(
                                example_hypothesis = example_hypothesis,
                                example_context_text = example_context_text,
                                example_proof = prompt_example['proof'],
                                example_reasoning_type = prompt_example['reasoning_type'],
                                test_hypothesis = hypothesis,
                                test_context_text = context_text)
    
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "user", "content": content},
                ])

    response = completion.choices[0].message

    response_text = response["content"]   #''.join(c for c in response)
    print(response_text)

    # print("Response\n",response)
    pred_proofs,pred_conclusion,pred_type = extract_answer(response_text)
    print("--------------------------------")
    print("Gold proofs",gold_proof_ls)
    print("Pred Proofs",pred_proofs)

    print("Gold_reasoning_type",gold_type_label)
    print("Pred_reasoning_type",pred_type)
    print("--------------------------------")

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

metric = evaluate(results,pred_typing = True)
result_dict['id-shot'] = metric


# ood-random-shot
results = []
save_path = f'./results/human_results/new_gpt_35_human_{test_type}_1_shot_ood.jsonl'

for ex in choiced_test_datas:

    prompt_example = random.choice(ood_dataset)

    hypothesis = normalize_sentence(ex["hypothesis"])
    context = ex["context"]
    context_text = serialize_context(context) if isinstance(context,dict) else context


    example_hypothesis = normalize_sentence(prompt_example["hypothesis"])
    example_context = prompt_example["context"]
    example_context_text = serialize_context(example_context) if isinstance(example_context,dict) else example_context


    gold_proof_ls,gold_conclusion,gold_type_label = parse_proof(ex)

    
    content = single_shot_prompt.format(
                                example_hypothesis = example_hypothesis,
                                example_context_text = example_context_text,
                                example_proof = prompt_example['proof'],
                                example_reasoning_type = prompt_example['reasoning_type'],
                                test_hypothesis = hypothesis,
                                test_context_text = context_text)
    
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "user", "content": content},
                ])

    response = completion.choices[0].message

    response_text = response["content"]   #''.join(c for c in response)
    print(response_text)

    # print("======================================")
    # print("Response\n",response)
    pred_proofs,pred_conclusion,pred_type = extract_answer(response_text)

    gold_proof_ls = list(set(gold_proof_ls))
    pred_proofs = list(set(pred_proofs))

    print("--------------------------------")
    print("Gold proofs",gold_proof_ls)
    print("Pred Proofs",pred_proofs)

    print("Gold_reasoning_type",gold_type_label)
    print("Pred_reasoning_type",pred_type)
    print("--------------------------------")


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

metric = evaluate(results,pred_typing = True)
result_dict['ood-random-shot'] = metric

print(test_path)
print(result_dict)


