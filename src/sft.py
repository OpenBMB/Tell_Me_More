import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import bmtrain as bmt
from functools import partial
import time
import os
import random
import json
from transformers import LlamaForCausalLM
from model_center.model import Llama
from model_center.tokenizer import LlamaTokenizer
from functools import partial
from dataset_wrapper import PromptIterableDataset, collator
import wandb
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import logging
import shutil
import numpy as np
import math
from IPython import embed
from datasets import load_dataset, load_from_disk
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_tokenizer(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return tokenizer


def get_model(args):
    model = Llama.from_pretrained(args.model_name_or_path)
    if args.load_ckpt is not None:
        logger.info(f"loading model from {args.load_ckpt}")
        bmt.load(model, os.path.join(args.load_ckpt, "pytorch_model.pt"))
    return model


def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(
        model.parameters(), 
        weight_decay=args.weight_decay, 
        eps=1e-5, 
        betas=(0.9, 0.95)
    )
    if args.load_ckpt is not None:
        file_name = os.path.join(args.load_ckpt, "optim.rank-{}.opt".format(bmt.rank()))
        logger.info(file_name)
        if os.path.exists(file_name):
            logger.info("start to load gradient ckpt {}".format(file_name))
            states = torch.load(file_name)
            optimizer.load_state_dict(states)

    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    if args.lr_decay_style == "linear":
        lr_scheduler = bmt.lr_scheduler.Linear(
            optimizer,
            start_lr=args.lr,
            warmup_iter=int(args.warmup_iters),
            end_iter=args.lr_decay_iters,
            num_iter=args.start_step,
        )
    elif args.lr_decay_style == "cosine":
        bmt.print_rank("use cosine")
        class Cosine(bmt.lr_scheduler.WarmupLRScheduler):
            def get_lr_warmup(self, num_iter) -> float:
                return self.start_lr * num_iter / self.warmup_iter
                
            def get_lr_decay(self, num_iter) -> float:
                progress = (num_iter - self.warmup_iter) / max(1, (self.end_iter - self.warmup_iter))
                return max(self.start_lr * 0.1, self.start_lr * (0.1 + 0.45 * (1.0 + math.cos(progress * math.pi))))

        lr_scheduler = Cosine(
            optimizer,
            start_lr=args.lr,
            warmup_iter=int(args.warmup_iters),
            end_iter=args.lr_decay_iters,
            num_iter=args.start_step,
        )

    elif args.lr_decay_style == "noam":
        logger.info("use noam")
        lr_scheduler = bmt.lr_scheduler.Noam(
            optimizer,
            start_lr=args.lr,
            warmup_iter=int(args.warmup_iters),
            end_iter=args.lr_decay_iters,
            num_iter=args.start_step,
        )
    else:
        raise NotImplementedError
    # if args.start_step != 0:
        # logger.info(f"loading scheduler from step {args.start_step}")
        # lr_scheduler.load_state_dict(torch.load(os.path.join(args.save_dir, f"ultrachat_{args.model}/step_{args.start_step}/scheduler.pt")))
    return lr_scheduler


def setup_model_and_optimizer(args):
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    # get the model
    model = get_model(args)
    bmt.synchronize()
    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    return tokenizer, model, optimizer, lr_scheduler


def initialize():
    # get arguments
    parser = argparse.ArgumentParser("")
    
    # model training arguments
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model_name_or_path", default='/data/private/hebingxiang/model_weights/llama-2-7b')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument("--batch_size_per_device", default=2, type=int)
    parser.add_argument("--logging_step", default=100, type=int)
    parser.add_argument("--save_step", default=50000, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--with_eval", action="store_true")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay rate")
    parser.add_argument("--loss_scale", type=float, default=6553600, help="loss scale")
    parser.add_argument("--train_iters", type=int, default=2000000)

    # data parameters
    parser.add_argument('--data_setting', type=str, help='MTSD or MTMD')
    parser.add_argument('--data_dir', type=str, help='The directory for saving the dataset')
    parser.add_argument('--max_train_samples', type=int, help='The maximum number of training samples')

    parser.add_argument('--cache_dir', type=str, help='The directory for cache')
    parser.add_argument("--save_dir", type=str, default="/data/models/chenyulin/ultrachat-llama")

    parser.add_argument("--save_limit", type=int, default=None, help="ckpt saved limit number")

    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument(
        "--lr_decay_style",
        type=str,
        default="cosine",
        choices=["constant", "linear", "cosine", "exponential", "noam"],
        help="learning rate decay function",
    )
    parser.add_argument("--lr_decay_iters", type=int, default=None, help="lr decay steps")
    parser.add_argument("--start_step", type=int, default=0, help="step to start or continue training")
    parser.add_argument("--tensorboard", type=str, default=None, help="whether using tb")
    parser.add_argument("--load_ckpt", type=str, default=None, help="resumed ckpt")

    args = parser.parse_args()
    # init bmt 
    bmt.init_distributed(seed=args.seed, zero_level=3)
    
    # wandb
    if args.wandb and bmt.rank() == 0:
        wandb.init()

    return args


def format_missing_details(i, detail):
    return f"- {detail['description']}: {', '.join(detail['options'])}\n"


def format_one(idx, thought, action, is_vague, missing_details):
    response = ""
    if action['role'] == "user":
        response = f"{action['content']}"
    else:
        if action['type'] == 'New':
            response = f"[INQUIRY THOUGHT] {action['thought']}\n[INQUIRY] {action['content']}"
        elif action['type'] == 'summary':
            response = f"[SUMMARY THOUGHT] {action['thought']}\n[SUMMARY] {action['content']}"

    is_vague_str = f"[INITIAL THOUGHT] {thought}"
    if idx == 0:
        if is_vague:
            details_str = " Some aspects of missing details and potential options are as follows:\n"
            for i, detail in enumerate(missing_details):
                details_str += format_missing_details(i, detail)
            # is_vague_str = "The task is vague."
            response = is_vague_str + details_str + response
        else:
            # is_vague_str = "The task is clear."
            response = is_vague_str + '\n' + response

    return response


def preprocess_data(data, args):
    task_description = """You are an agent trying to understand the user's goal and summarize it. Please first ask users for more specific details with options, and finally summarize the user's intention.
--- Step 1: initial thought generation ---
1. Generate [INITIAL THOUGHT] about if the task is vague or clear and why.
2. List the important missing details and some according options if the task is vague.
--- Step 2: inquiry for more information if vague ---
1. If the task is vague, inquire about more details with options according to the list in [INITIAL THOUGHT].
2. Think about what information you have and what to inquire next in [INQUIRY THOUGHT].
3. Present your inquiry with options for the user to choose after [INQUIRY], and be friendly.
4. You could repeat Step 2 multiple times (but less than 5 times), or directly skip Step 2 if the user task is clear initially.
--- Step 3: summarize the user's intention ---
1. Make the summary once the information is enough. You do not need to inquire about every missing detail in [INITIAL THOUGHT].
2. List all the user's preferences and constraints in [SUMMARY THOUGHT]. The number of points should be the same as rounds of chatting.
3. Give the final summary after [SUMMARY] with comprehensive details in one or two sentences."""
    first_query = f"{task_description}\n\nHere is the task:\n{data['task']}"
    sequences = [first_query]

    if args.data_setting == "MTSD":
        sequences.extend([
            format_one(idx, data['thought'], action, data['vague'], data['missing_details'])
            for idx, action in enumerate(data['actions'])
        ])
        return {"data": sequences}
    elif args.data_setting == "MTMD":
        dataset = []
        for idx, action in enumerate(data['actions']):
            sequences.append(format_one(idx, data['thought'], action, data['vague'], data['missing_details']))
            if action['role'] == 'assistant':
                dataset.append({"data": sequences.copy()})
        return dataset
    else:
        raise ValueError(f"Not supported type {args.data_setting}")


def load_raw_dataset(args):
    dataset = []
    with open(args.data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if args.data_setting == "MTSD":
                dataset.append(preprocess_data(data, args))
            elif args.data_setting == "MTMD":
                dataset.extend(preprocess_data(data, args))
            else:
                raise ValueError(f"Not supported type {args.data_setting}")

    random.shuffle(dataset)
    if args.max_train_samples is not None:
        dataset = dataset[:args.max_train_samples]
    return dataset


def finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset):
    # tensorboard
    if args.tensorboard is not None and bmt.rank() == 0:
        from torch.utils.tensorboard import SummaryWriter
        import distutils.version  # noqa: F401

        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard)
        writer = SummaryWriter(log_dir=args.tensorboard)
    
    logger.info(f"total training instance number: {len(dataset)}")
    # args.train_iters = args.epochs * (len(dataset) // (bmt.world_size() * args.batch_size_per_device) + 1 )

    # loss function and optimizer manager
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    optim_manager = bmt.optim.OptimManager(loss_scale=args.loss_scale)
    optim_manager.add_optimizer(optimizer, lr_scheduler)
    bmt.synchronize()
    
    bmt.print_rank("Model memory")
    bmt.print_rank(torch.cuda.memory_summary())

    avg_time_recorder = bmt.utils.AverageRecorder()
    avg_loss_recorder = bmt.utils.AverageRecorder()
    train_start_time = time.time()
    global_step = 0

    bmt.print_rank("Model memory")
    bmt.print_rank(torch.cuda.memory_summary())

    # save step_0
    save_dir = os.path.join(args.save_dir, f"step_{global_step}")
    os.makedirs(save_dir, exist_ok=True)
    bmt.save(model, os.path.join(save_dir, "pytorch_model.pt"))
    if bmt.rank() == 0:
        tokenizer.save_pretrained(save_dir)
    bmt.print_rank(f"model saved at {save_dir}")

    logger.info("split data for each process")
    data_per_gpu = len(dataset) // bmt.world_size()
    # shuffle dataset
    dataset = dataset[bmt.rank() * data_per_gpu: (bmt.rank() + 1) * data_per_gpu]
    bmt.print_rank("training on [%d, %d] of the dataset" % (bmt.rank() * data_per_gpu, (bmt.rank() + 1) * data_per_gpu))
    dataset = PromptIterableDataset(
        dataset, 
        tokenizer=tokenizer, 
        max_seq_length=args.max_seq_length, 
        teacher_forcing=True, 
        truncate_method="tail",
    )

    logger.info("entering trainig loop...")
    for epoch in range(args.epochs):

        logger.info("wrapping up data")
        dataloader = DataLoader(dataset, batch_size=args.batch_size_per_device, collate_fn=partial(collator, tokenizer))

        if global_step >= args.train_iters:
            break
        progress_bar = tqdm(range(len(dataloader)), disable=not bmt.rank()==0, desc=f"epoch {epoch}")
        logger.info(f"*******start {epoch} epoch training********")
        for step, inputs in enumerate(dataloader):
            if global_step < args.start_step:
                # print(f"skip step {global_step}!")
                global_step += 1
                progress_bar.update(1)
                continue
            st = time.time()

            with bmt.inspect.inspect_tensor() as inspector:
                for k in inputs:
                    inputs[k] = inputs[k].cuda()
                labels = inputs.pop("labels")
                logits = model(**inputs).logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, len(tokenizer))
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                # print("logits:", shift_logits[:5, :10])
                # print("labels:", shift_labels[:10])
                loss = loss_func(shift_logits, shift_labels)
                # print(f"rank: {bmt.rank()}, loss: {loss.item()}")
                # loss = output.loss
                # loss = loss_func(logits, labels)
            
                global_loss = bmt.sum_loss(loss).item()

                optim_manager.backward(loss)


                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
                    optim_manager.clip_grad_norm(optimizer.param_groups, max_norm=args.clip_grad)

                    optim_manager.step()
                    optim_manager.zero_grad()

            
            global_step += 1
            progress_bar.update(1)

            # record time and loss
            iteration_time = time.time() - st

            avg_time_recorder.record(iteration_time)
            if not np.isnan(global_loss):
                avg_loss_recorder.record(global_loss)

            # print time and loss
            if global_step % args.logging_step == 0:
                bmt.print_rank(
                    "| Iter: {:6d} | loss: {:.4f} average_loss: {:.4f} | lr: {:.4e} | time: {:.4f} seconds | total_time_passed: {:.4f} minutes".format(
                        global_step,
                        global_loss,
                        avg_loss_recorder.value,
                        lr_scheduler.current_lr,
                        avg_time_recorder.value,
                        (time.time() - train_start_time) / 60
                    )
                )
                if args.wandb and bmt.rank() == 0:
                    wandb.log({
                        "loss": global_loss,
                        "average_loss": avg_loss_recorder.value,
                        "lr": lr_scheduler.current_lr,
                    }, step=global_step)
                if args.tensorboard and bmt.rank() == 0:
                    writer.add_scalar("Loss/train", global_loss, global_step)
                    writer.add_scalar("average_Loss/train", avg_loss_recorder.value, global_step)
                    writer.add_scalar("lr/train", lr_scheduler.current_lr, global_step)
            
            if global_step == args.train_iters:
                break

    save_dir = os.path.join(args.save_dir, f"step_{global_step}")
    os.makedirs(save_dir, exist_ok=True)
    bmt.save(model, os.path.join(save_dir, "pytorch_model.pt"))
    if bmt.rank() == 0:
        tokenizer.save_pretrained(save_dir)
    bmt.print_rank(f"model saved at {save_dir}")


def main():
    args = initialize()
    dataset = load_raw_dataset(args)
    args.train_iters = min(args.epochs * (len(dataset) // (bmt.world_size() * args.batch_size_per_device) + 1), args.train_iters)
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset)


if __name__ == "__main__":
    main()