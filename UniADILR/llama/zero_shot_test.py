import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from typing import List, Optional
import fire
from llama import Llama, Dialog
import torch.distributed as dist
import os
import json
import random
from utils import *
import time
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# os.environ["NPROC_PER_NODE"] = '2'
# from transformers import LlamaModel

# print("init_process_group")
# dist.init_process_group(backend='gloo', init_method='env://172.18.11.1:3252',  world_size = 2)
# print("init_process_group done")
# print("启动成功")

from prompt import zero_shot_prompt,zero_prompt_with_descript,zero_prompt_13b,select_prompt

import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# bleurt_path = '/data/webw5/exp/BackChain/EntailmentBank/generate/bleurt-tiny-512'
# bleurt_tokenizer = AutoTokenizer.from_pretrained(bleurt_path)
# bleurt_scorer = AutoModelForSequenceClassification.from_pretrained(bleurt_path)
# bleurt_scorer = bleurt_scorer.to(torch.cuda.current_())

def extract_new(text):
    matches = re.findall(r'Sentence \d+', text)
    proof_list = [p.replace("Sentence ","sent") for p in matches]
    proof_list = list(set(proof_list))
    return proof_list,'none','none'


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


def main(
    ckpt_dir: str = '/data/webw5/exp/BackChain/API/llama2-7b',
    tokenizer_path: str = '/data/webw5/exp/BackChain/API/llama2-7b/tokenizer.model',
    temperature: float = 0.9,
    top_p: float = 0.9,
    max_seq_len: int = 1024,  #1024  最大输入长度限制，不能输入过长
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = 200,
    with_descript: bool = False

):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """

    print("start main")

    
    def get_response(content):
        if len(content) > max_seq_len:
            content = content[:max_seq_len]

        prompt = [[{"role": "user",
                "content":content  # 按照格式输入instruction和input,和template结合得到prompt
        }]]
        #print("Prompt",prompt)

        # inputs = tokenizer(prompt, return_tensors="pt")
        # input_ids = inputs["input_ids"].to(device)

        # if len(prompt) > 1000:
        #     return '[PROOF] = NONE; [TYPE] = NONE'

        results = generator.chat_completion(
                dialogs= prompt ,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
        #print("Results",results)
        return results[0]['generation']['content']
    

    # zero-shot的模板不需要示例，只需要test的hypothesis 和context
    if with_descript:
        prompt_template = zero_prompt_with_descript
    else:
        prompt_template = zero_shot_prompt

    print("start building llama2")

    # generator = LlamaModel.from_pretrained('/data/webw5/exp/BackChain/Llama-2-13b-chat-hf')
    

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        #model_parallel_size = 2
        
    )
    print("Llmma2 built")


    #现在只做了单一提示和测试
    test_paths = ['/data/webw5/exp/BackChain/reasoning_type_exp/data/syn_data/abduction/abduction_test.jsonl',
                # '/data/webw5/exp/BackChain/reasoning_type_exp/data/syn_data/induction/induction_test.jsonl',
                # '/data/webw5/exp/BackChain/reasoning_type_exp/data/syn_data/deduction/deduction_test.jsonl',
                # '/data/webw5/exp/BackChain/reasoning_type_exp/data/human_data/abduction.jsonl',
                # '/data/webw5/exp/BackChain/reasoning_type_exp/data/human_data/deduction.jsonl',
                # '/data/webw5/exp/BackChain/reasoning_type_exp/data/human_data/induction.jsonl'
                ]
    

    #test_path = ['/data/webw5/exp/BackChain/reasoning_type_exp/data/syn_data/abduction/abduction_test.jsonl']


    metrics = []
    for test_path in test_paths:
        
        test_datas = read_datas([test_path])

        # print(len(test_datas))
        test_datas = test_datas[:100]

        results = []

        start_time = time.time()

        for ex in test_datas:
            hypothesis = normalize_sentence(ex["hypothesis"])
            context = ex["context"]
            context_text = serialize_context(context) if isinstance(context,dict) else context

            gold_proof_ls,gold_conclusion,gold_type_label = parse_proof(ex)
            content = select_prompt.format(test_hypothesis = hypothesis,
                                            test_context_text = context_text)
            # if hasattr(torch.cuda, 'empty_cache'):

            response = get_response(content = content)
            print("======================================")
            print("Response\n",response)
            pred_proofs,pred_conclusion,pred_type = extract_new(response)
            print("Proofs",pred_proofs)
            print("gold",gold_proof_ls)
            print("Conclusion",pred_conclusion)
            print("Pred_reasoning_type",pred_type)
            print("======================================")
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        

            num = 0


            # while len(pred_proofs) == 0 and num < 5:

            #     response = get_response(content = revise_prompt)
            #     print(response)

            #     pred_proofs,pred_conclusion,pred_type = extract_answer(response)
            #     num += 1



            results.append({"original_ouput":response,
                            
                            "proof_gt":gold_proof_ls,
                            "conclusion_label":gold_conclusion,
                            "type_label":gold_type_label,
                            
                            "proof_pred":pred_proofs,
                            "conclusion_pred":pred_conclusion,
                            "type_pred":pred_type
                            })
            del response

        print("Consumed time", str(time.time() - start_time))

        # print([predicts])

        save_path = './deduction_results.jsonl'


        with open(save_path, 'w') as jsonl_file:
            for item in results:
                json_line = json.dumps(item) + '\n'  # 将每个对象转换为 JSON 格式并加上换行符
                jsonl_file.write(json_line)

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
        

    #直接evaluate并返回results

    # for dialog, result in zip(dialogs, results):
    #     for msg in dialog:
    #         print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    #     print(
    #         f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
    #     )
    #     print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)