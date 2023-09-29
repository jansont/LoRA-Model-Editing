import json
import warnings
from pathlib import Path
from torch.utils.data import Dataset

def create_lm_dataset(prompts, target, subject, tokenizer, args):
    target = target.strip() ; subject = subject.strip()
    dataset = []
    for prompt in prompts: 
        prompt = prompt.strip()
        subject_position = prompt.lower().find(subject.lower())
        target_position = prompt.lower().find(target.lower())
        predict_target = target_position >= subject_position
        
        predict_idx = target_position if predict_target else subject_position
        predict_len = len(target) if predict_target else len(subject)
        predict_str = target if predict_target else subject
        if predict_idx > 0: 
            context = prompt[:predict_idx]
            completion = prompt[predict_idx:]
            target_completion = prompt[predict_idx:predict_idx+predict_len]      
        else: 
            warnings.warn("The subject or target does not seem to be in the prompt.")
            tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False)
            split_size = max(int(len(tokenized_prompt) * args.completion_size), 1)
            context = tokenized_prompt[:-split_size]
            completion = tokenized_prompt[-split_size:]
            target_completion = completion
            
        if type(context)==list: 
            if len(context) == 0: 
                continue
            
        tokenized_context = tokenizer.encode(context, add_special_tokens=False)
        tokenized_full_completion = tokenizer.encode(completion, add_special_tokens=False)
        tokenized_target_completion = tokenizer.encode(target_completion, add_special_tokens=False)
        
        dataset.append({
            "context" : tokenized_context, 
            "full_completion" : tokenized_full_completion, 
            "target_completion" : tokenized_target_completion,
            "predict_target" : predict_target,
            "str_label" : predict_str
        })
    return dataset   



class LoraCompensationDataset(Dataset):
    def __init__(self, dataset_path, training=True, test_size=0.1): 
        self.test_size = test_size
        with open(dataset_path, "r") as json_file:
            loaded_data = json.load(json_file)
            
        training_prompts = [] ; test_prompts = []
        for samples in loaded_data: 
            samples = samples["training_prompts"]
            test_samples = samples[:int(len(samples) * test_size)]
            training_samples = samples[int(len(samples) * test_size):]
            
            training_prompts += training_samples
            test_prompts += test_samples
    
        self.training = training
        self.training_prompts = training_prompts
        self.test_prompts = test_prompts
    
    def __len__(self):
        if self.training: 
            return len(self.training_prompts)
        else: 
            return len(self.test_prompts)
    
    def __getitem__(self, idx):
        if self.training: 
            return self.training_prompts[idx]
        else: 
            return self.test_prompts[idx]
        
        

class CorrectionDataset(Dataset):
    def __init__(self, 
                 dataset_path: Path, 
                 use_chat_gpt: bool = False,):
        with open(dataset_path, "r") as json_file:
            loaded_data = json.load(json_file)
        self.dataset = loaded_data
        self.use_chat_gpt = use_chat_gpt
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> tuple[str, str, str, str, list[str], list[str]]:
        sample = self.dataset[idx]
        prompt = sample["requested_rewrite"]["prompt"]
        subject = sample["requested_rewrite"]["subject"]
        target = sample["requested_rewrite"]["target_true"]["str"]
        target_new = sample["requested_rewrite"]["target_new"]["str"]
        
        neighborhood_prompts = sample["neighborhood_prompts"]
        neighborhood_prompts = [prompt.format(subject) + " " + target for prompt in neighborhood_prompts]
        
        same_attribute_prompts = sample["attribute_prompts"]
        same_attribute_prompts = [prompt.format(subject) + " " + target_new for prompt in same_attribute_prompts]
                
        try: 
            training_prompts = sample["training_prompts"]
        except KeyError: 
            raise KeyError("This dataset does not have training prompts. Please use the chat gpt dataset.")
        
        reference_evaluation_prompts = [
            sentence.replace(target_new, target) for sentence in training_prompts
            ]
        reference_same_attribute_prompts = [
            sentence.replace(target_new, target) for sentence in same_attribute_prompts
            ]
        reference_neighborhood_prompts = [
            sentence.replace(target, target_new) for sentence in neighborhood_prompts
            ]
    
        return {
            "prompt" : prompt, 
            "subject" : subject, 
            "target" : target, 
            "target_new" : target_new, 
            "training_prompts" : training_prompts,
            "reference_evaluation_prompts" : reference_evaluation_prompts,
            "neighborhood_prompts" : neighborhood_prompts,
            "reference_neighborhood_prompts" : reference_neighborhood_prompts,
            "same_attribute_prompts" : same_attribute_prompts,
            "reference_same_attribute_prompts" : reference_same_attribute_prompts,
        }
