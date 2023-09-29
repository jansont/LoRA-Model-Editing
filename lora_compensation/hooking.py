import torch
import transformer_lens.utils as utils
from functools import partial


def unembedding_function(model, residual_stack, cache) -> float:
    #we are only interested in applying the layer norm of the final layer on the final token
    #shape: [74, 5, 10, 1280] = n_layers, prompts, tokens, d_model
    z = cache.apply_ln_to_stack(residual_stack, layer = -1)
    z = z @ model.W_U
    return z


def get_residual(model, cache, new_target_token_idx, target_token_idx, decomposed=True, return_labels=False): 
    if decomposed: 
        retval = cache.decompose_resid(layer=-1, mode="all", return_labels=return_labels)
    else: 
        retval = cache.accumulated_resid(layer=-1, return_labels=return_labels)
    if return_labels:
        residual, layer_names = retval
    else:
        residual = retval
    
    residual = unembedding_function(model, residual, cache)
    #shape: torch.Size([layer, batch, pos, vocab])
    residual = residual.permute(0,2,1,3)
    
    new_target_token_idx = new_target_token_idx.unsqueeze(dim=0).unsqueeze(dim=0)
    new_target_token_idx = new_target_token_idx.repeat(residual.shape[0], residual.shape[1], 1,1)
    
    target_token_idx = target_token_idx.unsqueeze(dim=0).unsqueeze(dim=0)
    target_token_idx = target_token_idx.repeat(residual.shape[0], residual.shape[1],1,1)
    
    #shape: [layer, tokens, prompts, 1]
    mle_residual_logits = (residual.gather(dim=-1, index=new_target_token_idx) - residual.mean(dim=-1, keepdim=True))
    target_residual_logits = (residual.gather(dim=-1, index=target_token_idx) - residual.mean(dim=-1, keepdim=True))
    mle_residual_logits = mle_residual_logits[:,-1,:,:].squeeze()
    target_residual_logits = target_residual_logits[:,-1,:,:].squeeze()
    
    mle_residual_logits = mle_residual_logits.mean(dim=-1)
    target_residual_logits = target_residual_logits.mean(dim=-1)
    
    
    if return_labels: 
        return layer_names, mle_residual_logits.to("cpu"), target_residual_logits.to("cpu")
    else:
        return mle_residual_logits.to("cpu"), target_residual_logits.to("cpu")
    


def patch_layer(corrupted_residual_component,hook,cache):
    corrupted_residual_component[:, :, :] = cache[hook.name][:, :, :]
    return corrupted_residual_component

def patch_position(corrupted_residual_component, hook,pos,cache):
    corrupted_residual_component[:, pos, :] = cache[hook.name][:, pos, :]
    return corrupted_residual_component

def patch_head_pattern(
    corrupted_head_pattern,
    hook, 
    head_index, 
    cache):
    corrupted_head_pattern[:, head_index, :, :] = cache[hook.name][:, head_index, :, :]
    return corrupted_head_pattern

def extract_logit(logits, new_target_token_idx, target_token_idx):     
    mle_logit = (logits.gather(dim=-1, index=new_target_token_idx) - logits.mean(dim=-1, keepdim=True))
    target_logit = (logits.gather(dim=-1, index=target_token_idx) - logits.mean(dim=-1, keepdim=True))
    mle_logit = mle_logit.mean(dim=0)
    target_logit = target_logit.mean(dim=0)
    return mle_logit.to("cpu"), target_logit.to("cpu")


def get_residuals_and_logits(
        model, 
        device,
        clean_prompt: str,
        corrupted_prompts : list[str],
        target: str, 
        target_new: str, 
        ablate_with_corrupted=True):
    torch.cuda.empty_cache()

    #-----------------------------prepare inputs--------------------------------------
    clean_tokens = model.to_tokens(clean_prompt, prepend_bos=True) 
    corrupted_tokens = model.to_tokens(corrupted_prompts, prepend_bos=True)
    assert clean_tokens.shape[-1] == corrupted_tokens.shape[-1]

    clean_tokens = clean_tokens.expand(corrupted_tokens.shape[0], -1)
    
    try:
        target_token = model.to_single_token(target)
    except: 
        target_token = model.to_tokens(target)[:,1]

    try: 
        new_target_token = model.to_single_token(target_new)
    except:
        new_target_token = model.to_tokens(target_new)[:,1]
        
    
    if ablate_with_corrupted:
        ablate_tokens = corrupted_tokens
        reference_tokens = clean_tokens
    else:
        ablate_tokens = clean_tokens
        reference_tokens = corrupted_tokens
        
    reference_logits, reference_cache = model.run_with_cache(reference_tokens, return_type="logits")
    ablate_logits, ablate_cache = model.run_with_cache(ablate_tokens, return_type="logits")

    # reference_logits = reference_logits.to("cpu")
    # ablate_logits = ablate_logits.to("cpu")
    
    mle_token = torch.argmax(reference_logits[:,-1,:], dim=-1)
    target_token = torch.ones_like(mle_token).long().to(device) * target_token
    target_token = target_token.unsqueeze(dim=-1)
    new_target_token = torch.ones_like(mle_token).long().to(device) * new_target_token
    new_target_token = new_target_token.unsqueeze(dim=-1)
    del mle_token

    # mle_token = mle_token.to("cpu")
    # target_token = target_token.to("cpu")

    reference_logits = reference_logits[:,-1,:]
    ablate_logits = ablate_logits[:,-1,:]
    
    reference_new_logit, reference_target_logit = extract_logit(reference_logits, new_target_token, target_token)
    ablate_new_logit, ablate_target_logit = extract_logit(ablate_logits, new_target_token, target_token)
    #---------------------------calculating base results---------------------------------------
    layer_names, ref_decomposed_residual_new, ref_decomposed_residual_target = get_residual(model, reference_cache, new_target_token, target_token, decomposed=True, return_labels=True)
    ablate_decomposed_residual_new, ablate_decomposed_residual_target = get_residual(model, ablate_cache, new_target_token, target_token, decomposed=True)
    
    torch.cuda.empty_cache()          
    return {
    "layer_names" : layer_names,   
    
    "reference_new_logit" : reference_new_logit.to("cpu"),
    "reference_target_logit":reference_target_logit.to("cpu"),
    
    "ablate_new_logit":ablate_new_logit.to("cpu"),
    "ablate_target_logit":ablate_target_logit.to("cpu"),
    
    "reference_decomposed_residual_new":ref_decomposed_residual_new.to("cpu"),
    "reference_decomposed_residual_target":ref_decomposed_residual_target.to("cpu"),
    
    "ablate_decomposed_residual_new":ablate_decomposed_residual_new.to("cpu"),
    "ablate_decomposed_residual_target":ablate_decomposed_residual_target.to("cpu"),
}
    