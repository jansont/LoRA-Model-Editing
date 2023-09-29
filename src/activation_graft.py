import torch
import warnings
from functools import partial
from src.model import LORAConfig
from lib.transformer_lens.HookedTransformer import HookedTransformer
from lib.transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
) 
from lib.transformer_lens.utilities import devices
from lib.transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
from lib.transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from abc import ABC, abstractmethod

def patch_layer(corrupted_residual_component,hook,cache): 
    corrupted_residual_component[:, :, :] = cache[hook.name][:, :, :]
    return corrupted_residual_component

class Graft(ABC):
    def __init__(
        self, 
        model: HookedTransformer,
        clean_prompt: str,
        corrupted_prompts: list[str],
        target: str,
        device: str,
        use_mle_token_graft: bool = False,
    ):
        self.model = model
        self.device = device
        self.clean_prompt = clean_prompt
        self.corrupted_prompts = corrupted_prompts
        self.target = target
        self.use_mle_token_graft = use_mle_token_graft
    
    def pad_from_left(
            self, 
            tokens : torch.tensor,
            maxlen:int
        ) -> torch.tensor:
        pad_token = self.model.tokenizer.pad_token_id
        padded_tokenized_inputs = torch.zeros(tokens.shape[0], maxlen)
        
        n_pads = maxlen - tokens.shape[-1]
        padded_tokenized_inputs[:,n_pads] = pad_token
        padded_tokenized_inputs[:,n_pads:] = tokens
        return padded_tokenized_inputs.long()

    def pad_to_same_length(
            self, 
            clean_tokens: torch.tensor,
            corrupted_tokens: torch.tensor
        ) -> tuple[torch.tensor, torch.tensor]: 
        maxlen = max([clean_tokens.shape[-1], corrupted_tokens.shape[-1]])
        
        if clean_tokens.shape[-1] > corrupted_tokens.shape[-1]: 
            corrupted_tokens = self.pad_from_left(corrupted_tokens, maxlen)
        elif clean_tokens.shape[-1] < corrupted_tokens.shape[-1]: 
            clean_tokens = self.pad_from_left(clean_tokens, maxlen)
        return clean_tokens, corrupted_tokens

    def unembedding_function(
            self, residual_stack: torch.tensor, cache, mlp=False) -> float:
        #we are only interested in applying the layer norm of the final layer on the final token
        #shape: [74, 5, 10, 1280] = n_layers, prompts, tokens, d_model
        z = cache.apply_ln_to_stack(residual_stack, layer = -1, mlp_input=mlp)
        z = z @ self.model.W_U
        return z

    def generate_lora_configs(self, args): 
        graft = self.graft
        layer_names = self.layer_names
        if "embed" in layer_names[0]:
            raise ValueError("Embeddings cannot be grafted")
        
        if args.task=="lora_graft_finetune":
            pass
                    
        elif args.task=="lora_mlp_finetune": 
            graft = torch.zeros_like(graft)
            for i, layer_name in enumerate(layer_names):
                if "mlp" in layer_name: 
                    graft[i] = 1
                    
        elif args.task=="lora_attn_finetune":
            graft = torch.zeros_like(graft)
            for i, layer_name in enumerate(layer_names):
                if "attn" in layer_name: 
                    graft[i] = 1
                    
        elif args.task=="lora_all_finetune":
            graft = torch.ones_like(graft)
        
        elif args.task=="finetune" or args.task=="graft_finetune":
            graft = torch.zeros_like(graft)
        else: 
            raise ValueError(f"Task {args.task} not recognized")
            
        lora_configs = []
        
        n_blocks = int(len(layer_names)/2)
        assert n_blocks%2==0 #we have 2 layers per block
        for block in range(n_blocks):
            attn_layer_name = layer_names[block*2]
            assert "attn" in attn_layer_name
            
            mlp_layer_name = layer_names[block*2+1]
            assert "mlp" in mlp_layer_name
            
            attn_graft_val = graft[block*2]
            mlp_graft_val = graft[block*2+1]
            
            layer_lora_configs = {}
            if attn_graft_val: 
                if not args.adapt_attn_c_attn and not args.adapt_attn_c_proj:
                    warnings.warn("No LORA config for attn layer despite grafting attn layer")
                layer_lora_configs["attn"] = (
                    LORAConfig(
                        layer=block,
                        layer_type="attn",
                        adapt_attn_c_attn=args.adapt_attn_c_attn,
                        adapt_attn_c_proj=args.adapt_attn_c_proj,
                        adapt_mlp_c_fc=False,
                        adapt_mlp_c_proj=False,
                        lora_dim=args.lora_dim,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout
                    )
                )
            else: 
                layer_lora_configs["attn"] = None
                
            if mlp_graft_val:
                if not args.adapt_mlp_c_fc and not args.adapt_mlp_c_proj:
                    warnings.warn("No LORA config for mlp layer despite grafting mlp layer")
                layer_lora_configs["mlp"] = (
                    LORAConfig(
                        layer=block,
                        layer_type="mlp",
                        adapt_attn_c_attn=False,
                        adapt_attn_c_proj=False,
                        adapt_mlp_c_fc=args.adapt_mlp_c_fc,
                        adapt_mlp_c_proj=args.adapt_mlp_c_proj,
                        lora_dim=args.lora_dim,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout
                    )
                ) 
            else:
                layer_lora_configs["mlp"] = None
                
            lora_configs.append(layer_lora_configs)   
        #checking the lora configs
        # combinations = [
        #     (l.layer, l.layer_type) for l in ]
        # ]
        # if has_duplicates(combinations):
        #     raise ValueError("Duplicate LORA configs found")
        return lora_configs
    
    def prepare_inputs(self): 
        clean_tokens = self.model.to_tokens(self.clean_prompt, prepend_bos=True) 
        corrupted_tokens = self.model.to_tokens(self.corrupted_prompts, prepend_bos=True)
        assert clean_tokens.shape[-1] == corrupted_tokens.shape[-1]
    
        clean_tokens = clean_tokens.expand(corrupted_tokens.shape[0], -1)
        target_token = self.model.to_single_token(self.target)
        
        corrupted_logits, corrupted_cache = self.model.run_with_cache(corrupted_tokens, return_type="logits")
        clean_logits, clean_cache = self.model.run_with_cache(clean_tokens, return_type="logits")
        
        mle_token = torch.argmax(clean_logits[:,-1,:], dim=-1)      
        target_token = torch.ones_like(mle_token).long().to(self.device) * target_token
        target_token = mle_token if self.use_mle_token_graft else target_token
        target_token = target_token.unsqueeze(dim=-1)
        
        clean_logits = clean_logits[:,-1,:]
        corrupted_logits = corrupted_logits[:,-1,:]
    
        clean_target_logit = (clean_logits.gather(dim=-1, index=target_token) - clean_logits.mean(dim=-1, keepdim=True)).to("cpu")
        clean_target_logit = clean_target_logit.mean(dim=0)
        
        corrupted_target_logit = (corrupted_logits.gather(dim=-1, index=target_token) - corrupted_logits.mean(dim=-1, keepdim=True)).to("cpu")
        corrupted_target_logit = corrupted_target_logit.mean(dim=0)

        return {
            "clean_cache" : clean_cache,
            "corrupted_cache" : corrupted_cache,
            "clean_target_logit" : clean_target_logit,
            "corrupted_target_logit" : corrupted_target_logit,
            "clean_tokens" : clean_tokens,
            "corrupted_tokens" : corrupted_tokens,
            "target_token" : target_token,
        }
    
        
class DecompositionGraft(Graft): 
    def __init__(self, 
                 model: HookedTransformer,
                 clean_prompt: str,
                 corrupted_prompts: list[str],
                 target: str, 
                 device,
                 use_mle_token_graft: bool = False, 
                 graft_threshold=0.75):
        super().__init__(model, clean_prompt, corrupted_prompts, target, device, use_mle_token_graft)
        self.graft_threshold=graft_threshold
        
    def run(self):
        inputs = self.prepare_inputs()
        clean_cache = inputs["clean_cache"]
        corrupted_cache = inputs["corrupted_cache"]
        target_token = inputs["target_token"]
  
        residual_clean_stack, layer_names = clean_cache.decompose_resid(layer=-1, return_labels=True)       
        residual_corrupted_stack = corrupted_cache.decompose_resid(layer=-1, return_labels=False)
    
        token_idx_expanded = target_token.repeat(residual_clean_stack.shape[0],1,1)
        
        residual_clean_stack = self.unembedding_function(residual_clean_stack, clean_cache)
        residual_clean_stack = residual_clean_stack[:,:,-1,:]
        residual_clean_logits = residual_clean_stack.gather(index=token_idx_expanded, dim=-1) - residual_clean_stack.mean(dim=-1, keepdim=True)
        
        residual_corrupted_stack = self.unembedding_function(residual_corrupted_stack, corrupted_cache)
        residual_corrupted_stack = residual_corrupted_stack[:,:,-1,:]
        residual_corrupted_logits = residual_corrupted_stack.gather(index=token_idx_expanded, dim=-1) - residual_clean_stack.mean(dim=-1, keepdim=True)
        
        direct_effect = (residual_clean_logits - residual_corrupted_logits)
        direct_effect = direct_effect.squeeze().mean(dim=-1)
        
        graft = direct_effect > (direct_effect.max() * self.graft_threshold)
        layer_names = layer_names[2:] #removing embeddings as we are not interested in finetuning these
        graft = graft[2:]
        self.graft = graft
        self.layer_names = layer_names
 
   
class CausalTotalEffectGraft(Graft): 
    def __init__(self, 
                 model: HookedTransformer,
                 clean_prompt: str,
                 corrupted_prompts: list[str],
                 target: str, 
                 device: str,
                 use_mle_token_graft: bool = False, 
                 graft_threshold=0.75):
        super().__init__(model, clean_prompt, corrupted_prompts, target, device, use_mle_token_graft)
        self.graft_threshold=graft_threshold
    
    def run(self): 
        inputs = self.prepare_inputs()
        clean_cache = inputs["clean_cache"]
        target_token = inputs["target_token"]      
        corrupted_target_logit = inputs["corrupted_target_logit"]
        corrupted_tokens = inputs["corrupted_tokens"] 
        
        #---------------------------calculating patch results---------------------------------------
        all_total_effects = torch.zeros(self.model.cfg.n_layers * 2, device="cpu", dtype=torch.float32)

        patched_layer_names = []
        for layer in range(self.model.cfg.n_layers*2):
            if layer % 2 == 0: 
                p = "attn_out"
            else: 
                p = "mlp_out"
            patched_layer_names.append(f"{layer//2}_{p}")
            patch_name = f"blocks.{layer//2}.hook_{p}"
            hook_fn = partial(patch_layer, cache=clean_cache)            
            with self.model.hooks(
                fwd_hooks = [(patch_name, hook_fn)]
            ) as hooked_model:
                restored_logits, _ = hooked_model.run_with_cache(corrupted_tokens, return_type="logits")
                restored_logits = restored_logits[:,-1,:]
                
                restored_target_logit = (restored_logits.gather(dim=-1, index=target_token) - restored_logits.mean(dim=-1, keepdim=True)).to("cpu")
                restored_target_logit = restored_target_logit.mean(dim=0)
                
                total_effect = (restored_target_logit - corrupted_target_logit).squeeze()
                all_total_effects[layer] = total_effect
                
        graft = all_total_effects > (all_total_effects.max() * self.graft_threshold)
        
        self.graft = graft
        self.layer_names = patched_layer_names

        
class CausalTotalEffectWindowGraft(Graft): 
    def __init__(self, 
                 model: HookedTransformer,
                 clean_prompt: str,
                 corrupted_prompts: list[str],
                 target: str, 
                 device: str,
                 use_mle_token_graft: bool = False, 
                 graft_threshold:float=0.75, 
                 window_size:int=5, 
                 window_stride:int=1):
        super().__init__(model, clean_prompt, corrupted_prompts, target, device, use_mle_token_graft)
        self.graft_threshold=graft_threshold
        self.window_size = window_size
        self.window_stride=window_stride
        
    def generate_sliding_windows(self, n):
        tensor = torch.arange(n)
        windows = [tensor[i:i+self.window_size] for i in range(0, len(tensor) - self.window_size + 1, self.window_stride)]
        return torch.stack(windows)
    
    def get_layer_names(self, layer, layer_type):
        patch_names_dict = {
            "attn" : f"blocks.{layer//2}.hook_attn_out",
            "mlp" : f"blocks.{layer//2}.hook_mlp_out"
        }
        return patch_names_dict[layer_type]
        
    def run(self): 
        inputs = self.prepare_inputs()
        clean_cache = inputs["clean_cache"]
        target_token = inputs["target_token"]      
        corrupted_target_logit = inputs["corrupted_target_logit"]
        corrupted_tokens = inputs["corrupted_tokens"] 
        
        #---------------------------calculating patch results---------------------------------------
        all_total_effects = torch.zeros(self.model.cfg.n_layers * 2, device="cpu", dtype=torch.float32)
        layer_part_of_ablation = torch.zeros(self.model.cfg.n_layers * 2, device="cpu", dtype=torch.int16)
        windows = self.generate_sliding_windows(self.model.cfg.n_layers*2)
        for layer in range(self.model.cfg.n_layers*2-self.window_size+1): 
            #for each layer, get the neighbouring layers in a window
            window_patches = windows[layer]
            patch_names = []
            for i in window_patches:
                i = i.item()
                layer_type = "attn" if i%2==0 else "mlp"
                patch_name = self.get_layer_names(i, layer_type)
                patch_names.append(patch_name)
            
            hook_fn = partial(patch_layer, cache=clean_cache)     
            fwd_hooks = [(l, hook_fn) for l in patch_names]
                        
            with self.model.hooks(
                fwd_hooks = fwd_hooks
            ) as hooked_model:
                #run the model on the corrupted tokens and restore the 
                restored_logits, _ = hooked_model.run_with_cache(corrupted_tokens, return_type="logits")
                restored_logits = restored_logits[:,-1,:]
                
                restored_target_logit = (restored_logits.gather(dim=-1, index=target_token) - restored_logits.mean(dim=-1, keepdim=True)).to("cpu")
                restored_target_logit = restored_target_logit.mean(dim=0)

                total_effect = (restored_target_logit - corrupted_target_logit).squeeze()
                all_total_effects[window_patches] += total_effect
                layer_part_of_ablation[window_patches] += 1
                
        all_total_effects = all_total_effects / layer_part_of_ablation  
        graft = all_total_effects > (all_total_effects.max() * self.graft_threshold)
        
        patched_layer_names = []
        for layer in range(self.model.cfg.n_layers*2):
            if layer % 2 == 0: 
                patched_layer_names.append(f"{layer//2}_attn_out")
            else:
                patched_layer_names.append(f"{layer//2}_mlp_out")
    
        self.graft = graft
        self.layer_names = list(patched_layer_names)


class CausalDirectEffectWindowGraft(Graft): 
    #TODO: finish implementing this
    def __init__(self, 
                 model: HookedTransformer,
                 clean_prompt: str,
                 corrupted_prompts: list[str],
                 target: str, 
                 device: str,
                 use_mle_token_graft: bool = False, 
                 graft_threshold=0.75, 
                 window_size=5):
        super().__init__(model, clean_prompt, corrupted_prompts, target, device, use_mle_token_graft)
        self.graft_threshold=graft_threshold
        self.window_size = window_size
        if self.window_size < 1: 
            raise ValueError("Window size must be at least 1")
    

        
    def run(self): 
        inputs = self.prepare_inputs()
        clean_cache = inputs["clean_cache"]
        corrupted_cache = inputs["corrupted_cache"]
        target_token = inputs["target_token"]      
        corrupted_target_logit = inputs["corrupted_target_logit"]
        corrupted_tokens = inputs["corrupted_tokens"] 
        
        residual_clean_stack = clean_cache.decompose_resid(layer=-1, return_labels=False)       
        residual_corrupted_stack = corrupted_cache.decompose_resid(layer=-1, return_labels=False)
        token_idx_expanded = target_token.repeat(residual_clean_stack.shape[0],1,1)
        
        residual_clean_stack = self.unembedding_function(residual_clean_stack, clean_cache)
        residual_clean_stack = residual_clean_stack[:,:,-1,:]
        residual_clean_logits = residual_clean_stack.gather(index=token_idx_expanded, dim=-1) - residual_clean_stack.mean(dim=-1, keepdim=True)
        residual_corrupted_stack = self.unembedding_function(residual_corrupted_stack, corrupted_cache)
        residual_corrupted_stack = residual_corrupted_stack[:,:,-1,:]
        residual_corrupted_logits = residual_corrupted_stack.gather(index=token_idx_expanded, dim=-1) - residual_clean_stack.mean(dim=-1, keepdim=True)
        
        direct_effect = (residual_clean_logits - residual_corrupted_logits).mean(dim=-1)
        
        graft = direct_effect > (direct_effect.max() * self.graft_threshold)
        
        #---------------------------calculating patch results---------------------------------------
        all_total_effects = torch.zeros(self.model.cfg.n_layers * 2, device="cpu", dtype=torch.float32)

        windows = self.generate_sliding_windows(self.model.cfg.n_layers*2)
        patched_layer_names = set()
        for layer in range(self.model.cfg.n_layers*2): 
            window_patches = windows[layer]
            patch_names_dict = {
                0 : f"blocks.{layer//2}.hook_attn_out",
                1 : f"blocks.{layer//2}.hook_mlp_out"
            }
            patch_names = [patch_names_dict[l%2] for l in window_patches]
            fwd_hooks = [(l, hook_fn) for l in patch_names]
            
            hook_fn = partial(patch_layer, cache=clean_cache)            
            with self.model.hooks(
                fwd_hooks = fwd_hooks
            ) as hooked_model:
                restored_logits, _ = hooked_model.run_with_cache(corrupted_tokens, return_type="logits")
                restored_logits = restored_logits[:,-1,:]
                
                restored_target_logit = (restored_logits.gather(dim=-1, index=target_token) - restored_logits.mean(dim=-1, keepdim=True)).to("cpu")
                restored_target_logit = restored_target_logit.mean(dim=0)

                total_effect = (restored_target_logit - corrupted_target_logit).squeeze()
                all_total_effects[layer] = total_effect
                patched_layer_names.add(set(patch_names))
                
        graft = all_total_effects > (all_total_effects.max() * self.graft_threshold)
        self.graft = graft
        self.layer_names = patched_layer_names    



def has_duplicates(tuple_list):
    seen_tuples = set()

    for tuple_elem in tuple_list:
        if tuple_elem in seen_tuples:
            return True
        seen_tuples.add(tuple_elem)

    return False

def get_mlp_weights(model,num_layers, hidden_dim):
  Ks = []
  Vs = []
  for j in range(num_layers):
    K = model.get_parameter(f"blocks.{j}.mlp.W_in").T.detach()
    V = model.get_parameter(f"blocks.{j}.mlp.W_out")
    Ks.append(K)
    Vs.append(V)
  
  Ks =  torch.cat(Ks)
  Vs = torch.cat(Vs)
  K_heads = Ks.reshape(num_layers, -1, hidden_dim)
  V_heads = Vs.reshape(num_layers, -1, hidden_dim)
  return K_heads, V_heads

def get_attention_heads(model, num_layers, hidden_dim, num_heads, head_size):
  Vs = []
  for j in range(num_layers):
    v = model.get_parameter(f"blocks.{j}.attn.W_V").detach().T
    v = v - torch.mean(v, dim=0) 
    Vs.append(v.T)

  W_V = torch.cat(Vs)
  W_O = torch.cat([model.get_parameter(f"blocks.{j}.attn.W_O") for j in range(num_layers)]).detach()
  W_V_heads = W_V.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
  W_O_heads = W_O.reshape(num_layers, num_heads, head_size, hidden_dim).permute(0, 1, 3, 2)
  return W_V_heads, W_O_heads


def cosine_sim(x,matrix):
    dots = []
    for i in range(matrix.shape[0]):
        y = matrix[i,:] 
        s = torch.dot(x,y) / (torch.norm(x) * torch.norm(y))
        dots.append(s)
    return torch.stack(dots)
      
def normalize_and_entropy(V, eps=1e-6):
    absV = torch.abs(V)
    normV = absV / torch.sum(absV)
    entropy = torch.sum(normV * torch.log(normV + eps)).item()
    return -entropy
      
class SVD_Graft(Graft): 
    def __init__(self, 
                 model: HookedTransformer,
                 clean_prompt: str,
                 corrupted_prompts: list[str],
                 target: str, 
                 device: str,
                 use_mle_token_graft: bool = False, 
                 num_layers = 12,
                num_heads = 12,
                head_size = 64,
                hidden_dim = 768,
                top_n_vect=10,
                 graft_threshold=0.15):
        super().__init__(model, clean_prompt, corrupted_prompts, target, device, use_mle_token_graft)
        self.graft_threshold=graft_threshold
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.hidden_dim = hidden_dim
        self.top_n_vect = top_n_vect
        
    def run(self): 
        #---------------------------calculating patch results---------------------------------------
        svd_graft = torch.zeros(self.model.cfg.n_layers * 2, device="cpu", dtype=torch.float32)
        
        prompt = self.clean_prompt + " " + self.target
        prompt_tokens = self.model.to_tokens(prompt, prepend_bos=True)
        embedding = self.model.embed(prompt_tokens).squeeze()
                    
        K,V = get_mlp_weights(self.model, num_layers = self.num_layers, hidden_dim = self.hidden_dim)
        W_V_heads, W_O_heads = get_attention_heads(self.model, num_layers=self.num_layers, hidden_dim=self.hidden_dim, num_heads=self.num_heads, head_size = self.head_size)

        for layer_idx in range(self.num_layers): 
            for head_idx in range(self.num_heads):
                W_V_tmp, W_O_tmp = W_V_heads[layer_idx, head_idx, :], W_O_heads[layer_idx, head_idx]
                OV = W_V_tmp @ W_O_tmp.T
                U,S,V = torch.linalg.svd(OV)
                V = V.squeeze()
                S_attn = S[:self.top_n_vect]
                Vs_attn = []
                for i in range(self.top_n_vect):
                    Sc = cosine_sim(V[i,:], embedding)
                    Vs_attn.append(Sc)
                            
            W_matrix = K[layer_idx, :,:]
            U,S,V = torch.linalg.svd(W_matrix,full_matrices=False)
            S_mlp = S[:self.top_n_vect]
            Vs_mlp = []
            for i in range(self.top_n_vect):
                Sc = cosine_sim(V[i,:], embedding)
                Vs_mlp.append(Sc)
                
            Vs_attn = torch.stack(Vs_attn)
            Vs_mlp = torch.stack(Vs_mlp)
            
            max_s_value_mlp = torch.max(S_mlp)
            max_s_value_attn = torch.max(S_attn)
            
            Vs_attn = torch.abs(Vs_attn // max_s_value_attn) * Vs_attn
            Vs_mlp = torch.abs(Vs_mlp // max_s_value_mlp) * Vs_mlp
            Vs_attn = Vs_attn.mean(dim=-1).mean(dim=0)
            Vs_mlp = Vs_mlp.mean(dim=-1).mean(dim=0)
            
            svd_graft[layer_idx] = Vs_attn > self.graft_threshold
            svd_graft[layer_idx+1] = Vs_mlp > self.graft_threshold

        self.graft = svd_graft
        layer_names = []
        for l in range(self.num_layers):
            layer_names.append(f"{l}_attn_out")
            layer_names.append(f"{l}_mlp_out")
        self.layer_names = layer_names


class CircuitGraft(Graft): 
    def __init__(self, 
                 model: HookedTransformer,
                 clean_prompt: str,
                 corrupted_prompts: list[str],
                 target: str, 
                 device: str,
                 use_mle_token_graft: bool = False, 
                 graft_threshold=0.75):
        super().__init__(model, clean_prompt, corrupted_prompts, target, device, use_mle_token_graft)
        self.graft_threshold=graft_threshold
    
    def run(self): 
        inputs = self.prepare_inputs()
        clean_cache = inputs["clean_cache"]
        target_token = inputs["target_token"]      
        corrupted_target_logit = inputs["corrupted_target_logit"]
        corrupted_tokens = inputs["corrupted_tokens"] 
        
        #---------------------------calculating patch results---------------------------------------
        patched_layer_names = []
        hook_names = []
        fwd_hooks = []
        for layer in range(self.model.cfg.n_layers*2):
            if layer % 2 == 0: 
                p = "attn_out"
            else: 
                p = "mlp_out"
            patched_layer_names.append(f"{layer//2}_{p}")
            patch_name = f"blocks.{layer//2}.hook_{p}"
            hook_fn = partial(patch_layer, cache=clean_cache)       
            hook = (patch_name, hook_fn) 
            with self.model.hooks(
                fwd_hooks = fwd_hooks + [hook]
            ) as hooked_model:
                restored_logits, _ = hooked_model.run_with_cache(corrupted_tokens, return_type="logits")
                restored_logits = restored_logits[:,-1,:]
                
                restored_target_logit = (restored_logits.gather(dim=-1, index=target_token) - restored_logits.mean(dim=-1, keepdim=True)).to("cpu")
                restored_target_logit = restored_target_logit.mean(dim=0)
                
                total_effect = (restored_target_logit - corrupted_target_logit).squeeze()
                if total_effect < self.graft_threshold: 
                    fwd_hooks.append(hook)
                
                hook_names.append((p, layer//2, total_effect < self.graft_threshold))
                    
        graft = torch.zeros(self.model.cfg.n_layers * 2, device="cpu", dtype=torch.float32)
        for p, layer, graft_val in hook_names:
            graft[layer] = graft_val
        
        self.graft = graft
        self.layer_names = patched_layer_names