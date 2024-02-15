


from pytorch_lightning.utilities.cli import LightningCLI
#pytorch_lightning.cli.CLI
from dataset import ProofDataModels
from model import ProofModel
# from common import *


# os.path.insert(0,"./")

from typing import * 

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: Any) -> None:

        parser.link_arguments("model.model_path", "data.model_path")
        parser.link_arguments("data.pred_type", "model.pred_type")
        parser.link_arguments("data.max_input_len", "model.max_input_len")
        #parser.add_argument("--backened", default=0, type=int)




def main() -> None:
    # dist.init_process_group(backend='gloo')
    cli = CLI(ProofModel, ProofDataModels, save_config_overwrite=True)
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,7"
    

    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"  #  TORCH_ PL_TORCH_DISTRIBUTED_BACKEND
    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    # os.environ['MASTER_ADDR'] = 'localhost' 
    # os.environ['MASTER_PORT'] = '3016'
    main()
