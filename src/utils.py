import os
import json
import datetime
import torch
import torch.nn as nn
from pathlib import Path
import pickle

class AverageMeter(object):
    """Computes and stores the average and current value
         Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def set_all_trainable(model: nn.Module): 
    for n,p in model.named_parameters():
        p.requires_grad = True
        
    for n,p in model.named_parameters():
        print(n, p.requires_grad)

        
def set_trainable_from_graft(model: nn.Module, graft: torch.tensor):
    
    layer_names = graft.layer_names
    graft_vals = graft.graft
    
    graft_layers = [layer_name for layer_name, graft_val \
                                in zip(layer_names, graft_vals) if graft_val>0]
     
    trainable_layers = []
    for layer in graft_layers: 
        layer_info = layer.split("_")
        layer_num = int(layer_info[0])
        layer_type = layer_info[1]
        
        if layer_type=="attn":
            trainable_layers.append(f"transformer.h.{layer_num}.attn.c_proj.weight")
            trainable_layers.append(f"transformer.h.{layer_num}.attn.c_proj.bias")
        elif layer_type=="mlp":
            trainable_layers.append(f"transformer.h.{layer_num}.mlp.c_fc.weight")
            trainable_layers.append(f"transformer.h.{layer_num}.bias")
        else: 
            pass
        
    for n,p in model.named_parameters():
        if n in trainable_layers:
            print(n)
            p.requires_grad = True    


def save_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, default=str)  # Use default=str to handle non-serializable objects

def log_experiment(all_prompts, all_target, all_target_new, all_evaluations, all_init_evaluations, args):

    experiment_dir = Path("experiments") / args.experiment_name
    
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    else:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M")
        new_directory_name = f"{os.path.basename(experiment_dir)}_{timestamp}"
        new_directory_path = os.path.join(os.path.dirname(experiment_dir), new_directory_name)
        os.makedirs(new_directory_path)
        
    args_dict = vars(args)
    save_json(args_dict, os.path.join(experiment_dir, "experiment_config.json"))
    
    delta_results = []
    for _eval, _init_eval in zip(all_evaluations, all_init_evaluations):
        res = {}
        for k in _eval.keys():
            res[k] = _eval[k] - _init_eval[k]
        delta_results.append(res)
        
        
    for i, (prompt, target, new_target) in enumerate(zip(all_prompts, all_target, all_target_new)):
        str_prompt = f"{prompt} : {target} -> {new_target}"
        all_evaluations[i]["prompt"] = str_prompt
        all_init_evaluations[i]["prompt"] = str_prompt
        delta_results[i]["prompt"] = str_prompt
        
    save_json(all_evaluations, os.path.join(new_directory_path, "experiment_results.json"))    
    save_json(all_init_evaluations, os.path.join(new_directory_path, "experiment_init_results.json"))  
    save_json(delta_results, os.path.join(new_directory_path, "experiment_delta_results.json"))  