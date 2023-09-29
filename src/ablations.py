import torch
from transformer_lens.HookedTransformer import HookedTransformer

def get_mask(prompt_tokens, subject_tokens): 
    A_flattened = prompt_tokens.view(-1)
    B_flattened = subject_tokens.view(-1)

    mask = torch.zeros_like(A_flattened, dtype=torch.bool)

    for i in range(len(A_flattened) - len(B_flattened) + 1):
        if torch.all(A_flattened[i:i+len(B_flattened)] == B_flattened):
            mask[i:i+len(B_flattened)] = True
            
    return mask

def resample_ablation(model, prompt, subject, target, n_noise_samples=20, temperature=0.85):
    subject = " " + subject
    subject_tokens = model.to_tokens(subject)[:,1:]
    n_subject_tokens = subject_tokens.shape[-1]
    prompt_start = prompt.split("{")[0]
    clean_fact = prompt.format(subject)
    
    fact_tokens = model.to_tokens(clean_fact)
    clean_subject_mask = get_mask(fact_tokens, subject_tokens)
    corrupted_facts = []        
    while len(corrupted_facts) < n_noise_samples: 
        generated = model.generate(prompt_start, max_new_tokens=n_subject_tokens,temperature=temperature,verbose=False)
        corrupted_subject = generated.split(prompt_start)[-1].strip()
        corrupted_fact = prompt.format(corrupted_subject)
        corrupted_subject_tokens = model.to_tokens(corrupted_subject)[:,1:]
        corrupted_fact_tokens = model.to_tokens(corrupted_fact)
        
        corrupted_mask = get_mask(prompt_tokens=corrupted_fact_tokens, subject_tokens=corrupted_subject_tokens)
        if corrupted_mask.shape != clean_subject_mask.shape: 
            continue
        if all(corrupted_mask==clean_subject_mask):
            corrupted_facts.append(corrupted_fact)

    return clean_fact, corrupted_facts, target


def resample_ablation_uniform(model: HookedTransformer,
                      prompt: str, 
                      subject: str,
                      target: str, 
                      n_noise_samples=20) -> tuple[str, list[str], str]:
    subject = " " + subject
    subject_tokens = model.to_tokens(subject)[:,1:]
    n_subject_tokens = subject_tokens.shape[-1]
    
    clean_fact = prompt.format(subject)
    fact_tokens = model.to_tokens(clean_fact)
    clean_subject_mask = get_mask(fact_tokens, subject_tokens)
    
    embedding = model.W_E
    corrupted_facts = []
    while len(corrupted_facts) < n_noise_samples: 
    # for noise_sample in range(n_noise_samples): 
        permutations = torch.randperm(embedding.size(0))[:clean_subject_mask.shape[-1]]        
        random_samples = embedding[permutations]
        random_samples = random_samples.unsqueeze(dim=1)
        random_embeddings = model.unembed(random_samples)
        random_tokens = torch.argmax(random_embeddings, dim=-1).squeeze()
        random_tokens = random_tokens * clean_subject_mask + fact_tokens
        random_string = model.to_string(random_tokens[:,1:])[0]
        if model.to_tokens(random_string).shape[-1]==clean_subject_mask.shape[-1]:
            corrupted_facts.append(random_string)
    return clean_fact, corrupted_facts, target

def noise_ablation(model, prompt, subject, target, n_noise_samples=5, vx=3, device="cuda"):
    subject_tokens = model.to_tokens(subject)
    
    #shape: batch, n_tokens, embedding_dim
    subject_embedding = model.embed(subject_tokens)
    _, n_tokens, embedding_dim = subject_embedding.shape
    
    #noise: N(0,v), v = 3*std(embedding)
    embedding = model.W_E
    v = vx*torch.std(embedding, dim=0) #for each v in V
    noise = torch.randn(
        (n_noise_samples, n_tokens, embedding_dim)
    ).to(device) + v
    
    subject_embedding_w_noise = subject_embedding + noise
    
    #shape: batch, n_tokens, vocab_size (logits)
    unembedded_subject = model.unembed(subject_embedding_w_noise)

    noisy_subject_tokens = torch.argmax(unembedded_subject, dim=-1)
    noisy_subject_str = [
        model.to_string(nst) for nst in noisy_subject_tokens
    ]
    true_prompt = prompt.format(subject)
    corrupted_prompts = [
        prompt.format(nss) for nss in noisy_subject_str
    ]
    return true_prompt, corrupted_prompts, target