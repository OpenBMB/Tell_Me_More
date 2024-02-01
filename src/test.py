import argparse
import torch
import random
import string
from tqdm import tqdm
from torch.utils.data import DataLoader
import bmtrain as bmt
import os
import json
import numpy as np
from torch import nn
from model_center.model import Llama
from model_center.tokenizer import LlamaTokenizer
from model_center.generation.llama import LlamaRandomSampling
from functools import partial
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import logging
from IPython import embed
import datasets
from datasets import load_dataset, load_metric, load_from_disk
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize():
    # get arguments
    parser = argparse.ArgumentParser("")
    # model arguments
    parser.add_argument("--model_name_or_path", default='/data/private/hebingxiang/model_weights/llama-2-7b')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument("--batch_size_per_device", default=2, type=int)

    parser.add_argument("--data_dir", type=str, default='/data/datasets/merged_P3_small')
    parser.add_argument('--max_target_length', type=int, help='The maximum total sequence length for target text after tokenization.')
    parser.add_argument('--max_train_samples', type=int, help='The maximum number of training samples to calculate train ACC')
    parser.add_argument('--max_test_samples', type=int, help='The maximum number of testing samples to calculate test ACC')

    parser.add_argument('--cache_dir', type=str, help='The directory for cache')
    parser.add_argument('--output_dir', type=str, help='The directory for output')

    parser.add_argument("--tensorboard", type=str, default=None, help="whether using tb")
    parser.add_argument("--load_ckpt", type=str, default=None, help="resumed ckpt")

    args = parser.parse_args()
    # init bmt 
    bmt.init_distributed(seed=args.seed, zero_level=3)

    return args


def get_tokenizer(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return tokenizer


def get_model(args):
    # model = BaseModel(args)
    # TODO:
    model = Llama.from_pretrained(args.model_name_or_path)
    if args.load_ckpt is not None:
        logger.info(f"loading model from {args.load_ckpt}")
        # model.load_state_dict(torch.load(os.path.join(args.save_dir, f"ultrachat_{args.model}/step_{args.start_step}/checkpoint.pt")))
        # bmt.load(model, args.load_ckpt)
        bmt.load(model, os.path.join(args.load_ckpt, "pytorch_model.pt"))

    return model


def preprocess_data(data, tokenizer):
    task_description = """You are an agent trying to understand user's goal and summarize it. Please first ask users for more specific details with options, and finally summarize the user's intention.
--- Step 1: initial thought generation ---
1. Generate [INITIAL THOUGHT] about if the task is vague or clear and why.
2. List the important missing details and some according options if the task is vague.
--- Step 2: inquiry for more information if vague ---
1. If the task is vague, inquiry more details with options according to the list in [INITIAL THOUGHT].
2. Think what information you have and what to inquiry next in [INQUIRY THOUGHT].
3. Present your inquiry with options for user to choose after [INQUIRY], and be friendly.
4. You could repeat Step 2 for multiple times (but less than 5 times), or directly skip Step 2 if user task is clear initially.
--- Step 3: summarize user's intention ---
1. Make the summary once information is enough. You do not need to inquiry every missing details in [INITIAL THOUGHT].
2. List all the user's preferences and constraints in [SUMMARY THOUGHT]. The number of points should be same as rounds of chatting.
3. Give final summary after [SUMAARY] with comprhensive details in one or two sentences."""

    # test_prompt = tokenizer.apply_chat_template(
    #     [{"role": "user", "content": f"{task_description}\n\nHere is the task:\n{data['task']}"}],
    #     tokenize=False
    # )
    test_prompt = f"<s>User: {task_description}\n\nHere is the task:\n{data['task']}\nAgent: "

    return {"prompt": test_prompt, "label": data["actions"]}


def load_raw_dataset(args, tokenizer):
    tmp_dataset = []
    with open(args.data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            tmp_dataset.append(json.loads(line))
    random.shuffle(tmp_dataset)
    if args.max_test_samples is not None:
        tmp_dataset = tmp_dataset[:args.max_test_samples]
    
    dataset = []
    for data in tmp_dataset:
        dataset.append(preprocess_data(data, tokenizer))

    return dataset


def setup_model_and_tokenizer(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args)
    return tokenizer, model


def generate_on_p3_or_flan(args, test_dataset, tokenizer, model):

    BeamGen = LlamaRandomSampling(model, tokenizer)
    bs = args.batch_size_per_device

    logger.info("split data for each process")
    data_per_gpu = len(test_dataset) // bmt.world_size()
    test_dataset = test_dataset[bmt.rank() * data_per_gpu: (bmt.rank() + 1) * data_per_gpu]

    logger.info("wrapping up data")

    # generating!
    test_results = []

    test_iters = len(test_dataset) // bs + 1

    for i in tqdm(range(test_iters)):
        data_list = test_dataset[i * bs: (i + 1) * bs]
        prompts = [data["prompt"] for data in data_list]
        labels = [data["label"] for data in data_list]
        if len(prompts) == 0: continue
        preds = BeamGen.generate(
            prompts,
            max_length=args.max_target_length,
            repetition_penalty=1.2,
            # temperature=0.9,
            # top_p=0.95,
            # top_k=40,
        )
        test_results.extend([{
            "prompt": prompt,
            "pred": pred,
            "label": label,
        } for prompt, pred, label in zip(prompts, preds, labels)])

    # save test results
    save_path = os.path.join(args.output_dir, args.load_ckpt.split("/")[-1])
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"saving results to {os.path.join(save_path, 'test_results.json')}")
    with open(os.path.join(save_path, "test_results.json"), "w", encoding='utf-8') as fout:
        save_test_results = {
            "cnt": len(test_results),
            "data": test_results,
        }
        fout.write(json.dumps(save_test_results, ensure_ascii=False, indent=4))


def main():
    args = initialize()
    tokenizer, model = setup_model_and_tokenizer(args)
    raw_datasets = load_raw_dataset(args, tokenizer)
    generate_on_p3_or_flan(args, raw_datasets, tokenizer, model)


if __name__ == "__main__":
    main()
