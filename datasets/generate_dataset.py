import json
import openai
import langchain

import os, sys
import glob
import random
from collections import Counter, OrderedDict
import numpy as np
import torch
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
            
        
MODEL_NAME = "gpt-3.5-turbo"
MAX_TOKENS = 4000
TEMPERATURE = 0.7
INSTRUCTION = """
Paraphrase the following fact {n} times: {fact}. 
Seperate each paraphrase by a new line. Do not include numbers to list the paraphrases.
"""

def generate_dataset(dataset_path, sample_indices, n_counter=100): 
    with open(dataset_path, "r") as json_file:
        loaded_data = json.load(json_file)
    dataset = loaded_data
    new_dataset = []
    
    try: 
        for i in sample_indices: 
            sample = dataset[i]
            prompt = sample["requested_rewrite"]["prompt"]
            subject = sample["requested_rewrite"]["subject"]
            target_new = sample["requested_rewrite"]["target_new"]["str"]
            new_fact = prompt.format(subject) + " " + target_new

            instruct = INSTRUCTION.format(n=n_counter, fact=new_fact)

            response = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo',
                        messages=[
                            {"role": "user", "content": instruct}],
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
            )
            response = response["choices"][0]["message"]["content"]
        
            training_prompts = response.split("\n")
            
            sample["training_prompts"] = training_prompts
            new_dataset.append(sample)
            print(i)
        
    except KeyboardInterrupt: 
        pass
        
    return new_dataset


def main():
    dataset_path = Path("fact_dataset.json")
    sample_indices = list(range(0,100))
    n_counter = 100
    new_dataset = generate_dataset(dataset_path, sample_indices=sample_indices, n_counter=n_counter)
    with open("chatgpt_fact_dataset.json", "w") as f:
        json.dump(new_dataset, f)
        
        
if __name__ == "__main__":
    main()