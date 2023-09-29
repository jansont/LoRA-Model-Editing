import time
import wandb
import math
import os
import torch
from torch.cuda import amp
from src.utils import AverageMeter
from src.training.evaluate import evaluate
import lib.loralib as lora

def optimizer_step(_loss, _optimizer, _model, _schedule, args, is_update=True):
    if args.fp16:
        with amp.scale_loss(_loss, _optimizer) as _scaled_loss:
            _scaled_loss.backward()
    else:
        _loss.backward()

    if is_update:
        if args.clip > 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(_optimizer), args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)

        _optimizer.step()        
        _optimizer.zero_grad()

    if _schedule is not None:
        _schedule.step()


def train_validate(
    model, 
    optimizer, 
    scheduler, 
    train_loader, 
    valid_loader, 
    init_train_logits,
    init_valid_logits,
    args, 
    train_step=0, 
    epoch=0, 
):
    model.train()
    avg_lm_loss = AverageMeter()

    print('start to train the model................', epoch)
    log_start_time = time.time()
    best_val_ppl = None

    # train_loader.sampler.set_epoch(epoch)

    for idx, data in enumerate(train_loader):
        data = {key: value for key, value in data.items()}
        
        data = data["normal"]

        _input = data['input'].to(args.device)
        _target = data['target'].to(args.device)
        _msk = data['mask'].to(args.device)
        _eval_msk = data['eval_mask'].to(args.device)
        _init_logits = init_train_logits[idx].to(args.device)
        _lm_logits, _lm_loss = model(
            _input, lm_labels=_target, lm_mask=_msk, eval_mask=_eval_msk, label_smooth=args.label_smooth,
            init_logits=_init_logits,use_kl_reg=args.use_kl_reg,args=args, 
        ) 

        _lm_loss = _lm_loss.mean() 
  

        train_step += 1
        is_update = True if train_step % args.grad_acc == 0 else False
        avg_lm_loss.update(_lm_loss.item())
        optimizer_step(
            _lm_loss/(args.grad_acc), optimizer, model, scheduler, args, is_update=is_update
        )
        
        if train_step % args.log_interval == 0: 
            elapsed = time.time() - log_start_time
            lr = optimizer.param_groups[0]['lr']
            log_str = f'| epoch {epoch:3d} step {train_step:>8d} | { idx + 1:>6d} batches | ' \
                      f'lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | ' \
                      f'loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | ' \
                      f'ppl {math.exp(avg_lm_loss.avg):5.2f}'
            
            if args.do_wandb: 
                wandb.log({
                    "epoch": epoch,
                    "step": train_step,
                    "lr": lr,
                    "loss": avg_lm_loss.val,
                    "avg_loss": avg_lm_loss.avg,
                    "ppl": math.exp(avg_lm_loss.avg),
                })
                

            if args.rank == 0: 
                print(log_str)
            log_start_time = time.time()
            avg_lm_loss.reset()
        
        if train_step % args.save_interval == 0: 
            if args.rank == 0:
                model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
                print('saving checkpoint', model_path)
                torch.save({'model_state_dict': lora.lora_state_dict(model)}, model_path)

        # evaluation interval
        if train_step % args.eval_interval == 0:
            eval_start_time = time.time()
            evaluation = evaluate(model=model, valid_loader=valid_loader, args=args, init_logits=init_valid_logits, use_kl_reg=args.use_kl_reg)
            
            valid_loss = evaluation["avg_lm_loss"]
            valid_ppl = evaluation["avg_lm_ppl"]
            

            if best_val_ppl is None or valid_ppl < best_val_ppl:
                best_val_ppl = valid_ppl
                
            log_str = f'| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | ' \
                      f'time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | ' \
                      f'valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} '
                      
            if args.do_wandb:
                wandb.log(evaluation)

            if args.rank == 0:
                print('-' * 100)
                print(log_str)
                print('-' * 100)

            model.train()

        if train_step == args.max_step:
            break

    if args.rank == 0 and args.save_model:
        model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
        print('saving checkpoint', model_path)
        torch.save({'model_state_dict': model.state_dict()}, model_path) 
    return train_step


def initial_logits(
    model, 
    loader, 
    args,
):

    # train_loader.sampler.set_epoch(epoch)
    logits = []
    for idx, data in enumerate(loader):
        data = {key: value for key, value in data.items()}
        
        data = data["normal"]

        _input = data['input'].to(args.device)
        _target = data['target'].to(args.device)
        _msk = data['mask'].to(args.device)
        _eval_msk = data['eval_mask'].to(args.device)

        _lm_logits, _ = model(
            _input, lm_labels=_target, lm_mask=_msk, eval_mask=_eval_msk, label_smooth=args.label_smooth,args=args
        ) 
        logits.append(_lm_logits)
    return logits
