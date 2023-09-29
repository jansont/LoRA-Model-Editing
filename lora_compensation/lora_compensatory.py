import os
import yaml
import wandb
import pickle
import json
import types
import torch
import random
import argparse
import warnings
import itertools
import warnings
import loralib as lora
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformer_lens import HookedTransformer
from gpt2_lora.data_utils import FT_Dataset
from gpt2_lora.model import GPT2LMModel, GPT2Config, LORAConfig
from gpt2_lora.training.train import train_validate
from gpt2_lora.correction_dataset import CorrectionDataset, create_lm_dataset 
import gpt2_lora.ablations as ablations
import gpt2_lora.activation_graft as activation_grafts
from gpt2_lora.training.optimizer import (
    create_optimizer_scheduler, 
    add_optimizer_params, 
    create_adam_optimizer_from_args
)
from gpt2_lora.exp_utils import create_exp_dir
from gpt2_lora.training.evaluate import evaluate
from gpt2_lora.utils import set_all_trainable, set_trainable_from_graft, AverageMeter, log_experiment
from sklearn.model_selection import train_test_split
from lora_compensation.hooking import get_residuals_and_logits
from timeout_decorator import timeout, TimeoutError


# influence model, calculate the influence score between two samples.
def print_args(args):
    if args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)
        
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def validate_args(args):
    if args.task not in ['lora_graft_finetune', 'lora_mlp_finetune', 'lora_attn_finetune', 'lora_all_finetune', 'finetune', 'graft_finetune']: 
        raise ValueError("task not recognized")
    if args.task=="lora_graft_finetune": 
        if sum([args.adapt_mlp_c_fc, args.adapt_mlp_c_proj, args.adapt_attn_c_attn, args.adapt_attn_c_proj]) == 0: 
            raise ValueError("No LoRA layers selected")
    if args.task=="lora_mlp_finetune": 
        if sum([args.adapt_mlp_c_fc, args.adapt_mlp_c_proj]) == 0: 
            raise ValueError("No LoRA MLP layers selected")
    if args.task=="lora_attn_finetune": 
        if sum([args.aadapt_attn_c_attn, args.adapt_attn_c_proj]) == 0: 
            raise ValueError("No LoRA Attention layers selected")
    if args.graft_type not in ["decomposition", "causal_total_effect", "causal_total_effect_window", "causal_direct_effect_window"]: 
        raise ValueError("graft_type not recognized")
    if args.ablation_method not in ["noise", "resample", "resample_uniform"]: 
        raise ValueError("ablation_method not recognized")
    
        
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch GPT2 with LORA from Activation Grafting')
    
    # Add a new argument for the config file
    parser.add_argument('--config', default="configs/config_lora_compensatory.yaml", type=str, help='Path to the YAML config file')
    add_optimizer_params(parser)
    
    args = parser.parse_args()

    # Load YAML configuration
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    # Set the configuration values as attributes of args
    for key, value in config.items():
        setattr(args, key, value)

    setattr(args, 'device', get_device())
    validate_args(args)
    print_args(args)
    return args

def generate_lora_configs(layer: int, n_layers: int, args : types.SimpleNamespace, adapt_attn=True):
    lora_configs = [
        {"attn" : None, "mlp" : None} for _ in range(n_layers)
    ]
    if layer is None:
        return lora_configs
    if adapt_attn:
        lora_configs[layer]["attn"] = LORAConfig(
                        layer=layer,
                        layer_type="attn",
                        adapt_attn_c_attn=args.adapt_attn_c_attn,
                        adapt_attn_c_proj=args.adapt_attn_c_proj,
                        adapt_mlp_c_fc=False,
                        adapt_mlp_c_proj=False,
                        lora_dim=args.lora_dim,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout)
    else: 
        lora_configs[layer]["mlp"] = LORAConfig(
                        layer=layer,
                        layer_type="mlp",
                        adapt_attn_c_attn=False,
                        adapt_attn_c_proj=False,
                        adapt_mlp_c_fc=args.adapt_mlp_c_fc,
                        adapt_mlp_c_proj=args.adapt_mlp_c_proj,
                        lora_dim=args.lora_dim,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout) 
    return lora_configs


def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def run_experiment(args): 
    
    if args.model_name == "gpt2-small":
            hf_model_name = "gpt2"
            n_layer = 12
            config = GPT2Config(
                n_embd=768, n_layer=n_layer, n_head=12, 
    )
    elif args.model_name == "gpt2-large":
            hf_model_name = args.model_name
            n_layer = 36
            config = GPT2Config(
                n_embd=1280, n_layer=n_layer, n_head=20, 
    )
    else: 
        raise ValueError("model_name not recognized")
    lora_configs = generate_lora_configs(None, n_layer, args)

    lm_net = GPT2LMModel(config, lora_configs)
    
    model = GPT2LMHeadModel.from_pretrained(hf_model_name)
    state_dict = model.state_dict()
    lm_net.load_weight(state_dict)   
    
    model = HookedTransformer.from_pretrained(
        args.model_name, 
        hf_model=lm_net,
        lora_case=True
    )
    correction_dataset = CorrectionDataset(args.fact_data)
    
    #start by getting the base results
    correction_dataloader = DataLoader(correction_dataset, batch_size=1)
    initial_results = []
    print("..getting initial results")
    for batch_idx, batch in enumerate(correction_dataloader):
        #----------------------------Prepare Correction Dataset-----------------------------#
        prompt = batch["prompt"][0]
        subject = batch["subject"][0]
        target = batch["target"][0]
        target_new = batch["target_new"][0]
        training_prompts = [p[0] for p in batch["training_prompts"]]
        
        @timeout(30)
        def timeout_resample(ablation_method):
            if ablation_method == "resample_uniform": 
                original_fact, corrupted_facts, _ = ablations.resample_ablation_uniform(model, prompt,subject,target,                                                             n_noise_samples=args.noise_samples)
            elif ablation_method=="resample":
                original_fact, corrupted_facts, _ = ablations.resample_ablation(model, prompt, subject, target, n_noise_samples=args.noise_samples, temperature=args.temperature)
            elif ablation_method=="noise": 
                original_fact, corrupted_facts, _ = ablations.noise_ablation(model, prompt,subject,target,n_noise_samples=args.noise_samples)
            else: 
                raise ValueError("ablation_method not recognized")
            return original_fact, corrupted_facts
        
        
        try:
            original_fact, corrupted_facts = timeout_resample(args.ablation_method)
        except TimeoutError:
            warnings.warn(f"Resample timed out for prompt {prompt}")
            continue

        
        res = get_residuals_and_logits(model, 
                                args.device,
                                clean_prompt=original_fact,
                                corrupted_prompts=corrupted_facts,
                                target=target, 
                                target_new=target_new, 
                                ablate_with_corrupted=True)
        initial_results.append(res)
    save_pickle(initial_results, f"lora_compensation/results/{args.experiment_name}_initial_results.pkl")
            

    for layer in range(n_layer):
        if layer*2 % 2 == 0:
            adapt_attention = True
        else: 
            adapt_attention = False
            
        lora_configs = generate_lora_configs(layer, n_layer, args, adapt_attn=adapt_attention)
            
        lm_net = GPT2LMModel(config, lora_configs)
        model = GPT2LMHeadModel.from_pretrained(hf_model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(hf_model_name)
        state_dict = model.state_dict()
        lm_net.load_weight(state_dict)  
        
        
        correction_dataset = CorrectionDataset(args.fact_data)
        correction_dataloader = DataLoader(correction_dataset, batch_size=1)
        
        all_dataset = [] ; all_dataset_ref = []
        for batch_idx, batch in enumerate(correction_dataloader):
            #----------------------------Prepare Correction Dataset-----------------------------#
            prompt = batch["prompt"][0]
            subject = batch["subject"][0]
            target = batch["target"][0]
            target_new = batch["target_new"][0]
            training_prompts = [p[0] for p in batch["training_prompts"]]
            
            dataset = create_lm_dataset(
                    prompts=all_dataset, target=target_new,
                    subject=subject, tokenizer=tokenizer, args=args
                )
            dataset_ref = create_lm_dataset(
                prompts=all_dataset_ref, target=target,
                subject=subject, tokenizer=tokenizer, args=args
            )
            all_training_samples += dataset
            all_training_samples_ref += dataset_ref
            
        dataset_indices = list(range(len(dataset)))
        training_indices, valid_indices = train_test_split(
            dataset_indices, test_size=args.test_size, random_state=args.random_seed
        )
        training_prompts = [d for i,d in enumerate(all_dataset) if i in training_indices]
        valid_prompts = [d for i,d in enumerate(all_dataset) if i in valid_indices]
        training_prompts_ref = [d for i,d in enumerate(all_dataset_ref) if i in training_indices]
        valid_prompts_ref = [d for i,d in enumerate(all_dataset_ref) if i in valid_indices]
        
    
        train_data = FT_Dataset(
            samples=training_prompts,
            ref_samples=training_prompts_ref,
            batch_size=args.train_batch_size,
            max_seq_length=args.seq_len, 
            joint_lm=args.obj=='jlm'
        ) 
        valid_data = FT_Dataset(
            samples=valid_prompts,
            ref_samples=valid_prompts_ref,
            batch_size=args.train_batch_size,
            max_seq_length=args.seq_len, 
            joint_lm=args.obj=='jlm'
        )     
        train_loader = DataLoader(
            train_data, batch_size=args.train_batch_size, num_workers=0, 
            shuffle=False, pin_memory=False, drop_last=True,
        )
        valid_loader = DataLoader(
            valid_data, batch_size=args.valid_batch_size, num_workers=0, 
            shuffle=False, pin_memory=False, drop_last=False,
        )
        
        #---------------------------------Training Model------------------------------------#
        
        if args.fp16:
            try:
                from torch.cuda import amp
            except Exception as e:
                warnings.warn('Could not import amp, apex may not be installed')
        if args.max_step is None:
            args.max_step = (args.max_epoch * train_data.num_batches) 
            print('set max_step:', args.max_step)
        scheduler = create_optimizer_scheduler(optimizer, args)
        if args.fp16:
            lm_net, optimizer = amp.initialize(lm_net, optimizer, opt_level="O1")
        try:
            train_step = 0
            for epoch in itertools.count(start=1):
                train_step = train_validate(
                    lm_net, optimizer, scheduler, train_loader, valid_loader, args, 
                    train_step=train_step, epoch=epoch
                )
                print("REACH 5")
                
                if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
                    if args.rank == 0:
                        print('-' * 100)
                        print('End of training')
                    break
        except KeyboardInterrupt:
            if args.rank == 0:
                print('-' * 100)
                print('Exiting from training early')
            early_exit = True
            

        #--------------------Hooking model------------------
        model = HookedTransformer.from_pretrained(
            args.model_name, 
            hf_model=lm_net,
            lora_case=True
        )
        

    correction_dataloader = DataLoader(correction_dataset, batch_size=1)
    results = []
    for batch_idx, batch in enumerate(correction_dataloader):
        #----------------------------Prepare Correction Dataset-----------------------------#
        prompt = batch["prompt"][0]
        subject = batch["subject"][0]
        target = batch["target"][0]
        target_new = batch["target_new"][0]
        training_prompts = [p[0] for p in batch["training_prompts"]]
        
        @timeout(30)
        def timeout_resample(ablation_method):
            if ablation_method == "resample_uniform": 
                original_fact, corrupted_facts, _ = ablations.resample_ablation_uniform(model, prompt,subject,target,                                                             n_noise_samples=args.noise_samples)
            elif ablation_method=="resample":
                original_fact, corrupted_facts, _ = ablations.resample_ablation(model, prompt, subject, target, n_noise_samples=args.noise_samples, temperature=args.temperature)
            elif ablation_method=="noise": 
                original_fact, corrupted_facts, _ = ablations.noise_ablation(model, prompt,subject,target,n_noise_samples=args.noise_samples)
            else: 
                raise ValueError("ablation_method not recognized")
            return original_fact, corrupted_facts
        
        try:
            original_fact, corrupted_facts = timeout_resample(args.ablation_method)
        except TimeoutError:
            warnings.warn(f"Resample timed out for prompt {prompt}")
            continue

        res = get_residuals_and_logits(model, 
                                args.device,
                                clean_prompt=original_fact,
                                corrupted_prompts=corrupted_facts,
                                target=target, 
                                target_new=target_new, 
                                ablate_with_corrupted=True)
        results.append(res)
        module = "attn" if adapt_attention else "mlp"
    save_pickle(results, f"lora_compensation/results/{args.experiment_name}_layer_{layer}_{module}_results.pkl")
            
        
        


if __name__ == '__main__':
    
    args = parse_args()
    if args.do_wandb: 
        wandb.login()
        
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)   

    run_experiment(args)
    

    # #Sample code to run multiple experiments
    # variable_of_interest = "task"
    # variable_values = ["lora_graft_finetune", "lora_mlp_finetune"]
    
    # for experiment_idx in range(len(variable_values)): 
    #     args.variable_of_interest = variable_values[experiment_idx]
    #     args.experiment_name = f"{args.experiment_name}_{args.variable_of_interest}"
    #     run_experiment(args)
    

    