# import sys
# sys.path.insert(0,"../../type")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from model import ProofModel
from utils import read_datas
from torch.utils.data import Dataset, DataLoader
from dataset import EntireProofDataset,ProofDataModels

import pytorch_lightning as pl

# model_path  =  '/root/exp/model/t5-large',
# model = ProofModel(

#     model_path = '/root/exp/model/t5-large',
#     lr = 5e-4,
#     warmup_steps = 500,
#     num_beams = 10,
#     topk = 10,
#     max_input_len = 1024,
#     pred_type = False
# )  

#ckpt_path = '/data1/webw5/t5-3b/lightning_logs/version_2/checkpoints/epoch=59-step=10680.ckpt'

ckpt_path = '/data1/webw5/t5-3b/lightning_logs/version_8/checkpoints/epoch=14-step=2670.ckpt'
# mdoel = model.load_from_checkpoint(ckpt_path)

model = ProofModel.load_from_checkpoint(ckpt_path)

# test_settings = [
#     ['/data/webw5/exp/BackChain/reasoning_type_exp/data/syn_data/abduction/abduction_test.jsonl'],
#     ['/data/webw5/exp/BackChain/reasoning_type_exp/data/syn_data/deduction/deduction_test.jsonl'],
#     ['/data/webw5/exp/BackChain/reasoning_type_exp/data/syn_data/induction/induction_test.jsonl'],
#     # ['/data/webw5/exp/BackChain/reasoning_type_exp/data/human_data/abduction.jsonl'],
#     # ['/data/webw5/exp/BackChain/reasoning_type_exp/data/human_data/induction.jsonl'],
#     ['/data/webw5/exp/BackChain/reasoning_type_exp/data/human_data/abduction.jsonl'],
#     ['/data/webw5/exp/BackChain/reasoning_type_exp/data/human_data/deduction.jsonl'],
#     ['/data/webw5/exp/BackChain/reasoning_type_exp/data/human_data/induction.jsonl']
# ]


test_settings = [

    # ['/data/webw5/exp/BackChain/reasoning_type_exp/data/abduction_test/abduction_test.jsonl'],
    # ['/data/webw5/exp/BackChain/reasoning_type_exp/data/deduction_large/deduction_test.jsonl'],
    ['/data/webw5/exp/BackChain/reasoning_type_exp/man_data/deduction_new_complete.jsonl']
]
# test_settings = [
#     ['/data/webw5/exp/BackChain/reasoning_type_exp/data/deduction_large/deduction_test.jsonl'],

# ]

test_setting_names= ["在deduction上测试","在abduction上测试","在人工数据上测试"]

# datasets = []
# for setting in test_settings:
#     datasets.append( read_datas(setting))

# data_loaders = []
# for data in datasets:

#     dataset = EntireProofDataset( # type: ignore
#                             data,
#                             model_pth ='/root/exp/model/t5-large',
#                             max_input_len = 1024,
#                             max_output_len = 600,
#                             pred_type = False,
#                             is_train = False)
    
#     data_loaders.append(
#         DataLoader(
#             dataset,
#             batch_size = 4,
#             shuffle=False,
#             num_workers=0,
#             collate_fn=dataset.collate,
#             pin_memory=True,
#             drop_last=False
#         ))



data_modules = []

for test_setting in test_settings:
        
    data_modules.append(
        ProofDataModels(
            model_path=  '/data1/webw5/model/t5-3B',
            max_input_len= 1024,
            max_output_len= 600,
            batch_size= 1,
            num_workers= 0,
            train_path= None,
            val_path = None,
            test_path= test_setting,
            pred_type= False))
    
trainer = pl.Trainer(accelerator='gpu', 
                    devices=1)

for test_module in data_modules:
    trainer.test(model,test_module)

# for test_loader in data_loaders:
#     trainer.test(mdoel,test_loader)