import os
import yaml
import torch
import random
import argparse
import warnings
import itertools
import loralib as lora
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from transformer_lens import HookedTransformer
from gpt2_lora.data_utils import FT_Dataset
from gpt2_lora.model import GPT2LMModel, GPT2Config, LORAConfig
from gpt2_lora.training.train import train_validate
from gpt2_lora.correction_dataset import CorrectionDataset
from gpt2_lora.ablations import noise_ablation, resample_ablation
from gpt2_lora.activation_graft import CausalGraft, DecompositionGraft
from gpt2_lora.training.optimizer import (
    create_optimizer_scheduler, 
    add_optimizer_params, 
    create_adam_optimizer_from_args
)
from gpt2_lora.exp_utils import create_exp_dir
from sklearn.model_selection import train_test_split


# influence model, calculate the influence score between two samples.
def print_args(args):
    if args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch GPT2 with LORA from Activation Grafting')
    
    # Add a new argument for the config file
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    add_optimizer_params(parser)
    
    args = parser.parse_args()

    # Load YAML configuration
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    # Set the configuration values as attributes of args
    for key, value in config.items():
        setattr(args, key, value)

    print_args(args)

    return args



if __name__ == '__main__':
    args = parse_args()

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)    


    lora_configs = [
        # LORAConfig(
        #         layer=10,
        #         adapt_mlp=True,
        #         lora_dim=args.lora_dim,
        #         lora_alpha=args.lora_alpha,
        #         lora_dropout=args.lora_dropout)
        ]
    
    if args.model_name == "gpt2-small":
        hf_model_name = "gpt2"
        n_layer = 12
        config = GPT2Config(
            n_embd=768, n_layer=n_layer, n_head=12, 
        )
    elif args.model_name == "gpt2-large":
        hf_model_name = args.model_name
        n_layer = 35
        config = GPT2Config(
            n_embd=1280, n_layer=n_layer, n_head=20, 
        )
    else: 
        raise ValueError("model_name not recognized")
    
    lm_net = GPT2LMModel(config, lora_configs)
    
    model = GPT2LMHeadModel.from_pretrained(hf_model_name)
    state_dict = model.state_dict()
    lm_net.load_weight(state_dict)   
    
    model = HookedTransformer.from_pretrained(
        args.model_name, 
        hf_model=lm_net,
        lora_case=True
    )

    # model = HookedTransformer.from_pretrained(

    #         center_unembed=True,  
    #         center_writing_weights=True,              # Whether to center weights writing to the residual stream (ie set mean to be zero). Due to LayerNorm this doesn't change the computation.      
    #         fold_ln=True,                             # Whether to  fold in the LayerNorm weights to the subsequent linear layer.
    #         refactor_factored_attn_matrices=True,
    #     )