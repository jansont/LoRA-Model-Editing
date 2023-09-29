import time
import math
import torch
from src.utils import AverageMeter
from torch.nn.functional import softmax

def get_scores(model, data, init_logits, use_kl_reg, args): 
    _input = data["input"].to(args.device)
    _target = data["target"].to(args.device)
    _mask = data["mask"].to(args.device)
    _eval_mask = data["eval_mask"].to(args.device)
    _predict_target = data["predict_target"]
    _str_label = data["str_label"]
    init_logits = init_logits.to(args.device) if init_logits is not None else None

    lm_logits, _loss, = model(
        _input, lm_labels=_target, lm_mask=_mask, is_report_accuracy=False, init_logits=init_logits, use_kl_reg=use_kl_reg, 
        args=args) 
    
    loss = _loss.mean()
    _target = _target.unsqueeze(dim=-1)
    
    _target_logits = lm_logits.gather(dim=-1, index=_target)
    target_logits = _target_logits.squeeze(dim=-1) * _eval_mask
    
    probabilities = softmax(lm_logits, dim=-1) 
    target_probabilities = probabilities.gather(dim=-1, index=_target)
    target_probabilities = target_probabilities.squeeze(dim=-1) * _eval_mask
    
    target_logits = target_logits.sum()
    target_probabilities = target_probabilities.sum()
    return loss,target_logits, target_probabilities
    

def evaluate(model, valid_loader, init_logits,args, use_kl_reg=False,tokenizer=None):
    model.eval()
    total_loss = 0.
    start_time = time.time()

    avg_lm_loss = AverageMeter()
    avg_ref_loss = AverageMeter()
    avg_delta_loss = AverageMeter()
    avg_prob = AverageMeter()
    avg_ref_prob = AverageMeter()
    avg_delta_prob = AverageMeter()
    avg_logit = AverageMeter()
    avg_ref_logit = AverageMeter()
    avg_delta_logit = AverageMeter()

    with torch.no_grad():
        for idx, all_data in enumerate(valid_loader):

            data = {key: value for key, value in all_data["normal"].items()}
            reference_data = {key: value for key, value in all_data["reference"].items()}

            init_logits_ = init_logits[idx] if init_logits is not None else None
            loss,target_logits, target_probabilities = get_scores(model=model, data=data, args=args, use_kl_reg=use_kl_reg, init_logits=init_logits_)
            ref_loss, ref_target_logits, ref_target_probabilities = get_scores(model=model, data=reference_data, args=args, use_kl_reg=use_kl_reg, init_logits=init_logits_)

            delta_loss = loss - ref_loss
            delta_logit  = target_logits - ref_target_logits
            delta_prob = target_probabilities - ref_target_probabilities
            
            avg_lm_loss.update(loss.item())
            avg_ref_loss.update(ref_loss.item())
            avg_delta_loss.update(delta_loss.item())
            avg_prob.update(target_probabilities.item())
            avg_ref_prob.update(ref_target_probabilities.item())
            avg_delta_prob.update(delta_prob.item())
            avg_logit.update(target_logits.item())
            avg_ref_logit.update(ref_target_logits.item())
            avg_delta_logit.update(delta_logit.item())

        print('average loss', avg_lm_loss.avg)
    return {
        "avg_lm_loss" : avg_lm_loss.avg,
        "avg_ref_loss" : avg_ref_loss.avg,
        "avg_delta_loss" : avg_delta_loss.avg,
        "avg_prob" : avg_prob.avg,
        "avg_ref_prob" : avg_ref_prob.avg,
        "avg_delta_prob" : avg_delta_prob.avg,
        "avg_logit" : avg_logit.avg,
        "avg_ref_logit" : avg_ref_logit.avg,
        "avg_delta_logit" : avg_delta_logit.avg,
        "avg_lm_ppl" : math.exp(avg_lm_loss.avg),
        "avg_ref_ppl" : math.exp(avg_ref_loss.avg),
        "avg_delta_ppl" : math.exp(avg_delta_loss.avg),
    }

