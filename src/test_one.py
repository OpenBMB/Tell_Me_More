import argparse
import bmtrain as bmt
import os
import copy
import json
from model_center.model import Llama
from model_center.tokenizer import LlamaTokenizer
from model_center.generation.llama import LlamaRandomSampling
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYS_PROMPT = """You are an agent trying to understand user's goal and summarize it. Please first ask users for more specific details with options, and finally summarize the user's intention.
--- Step 1: inquiry for more information if necessary ---
1. If the user is unclear, ask for clarification.
2. Provide some options when querying.
3. Please only ask one question in one round of conversation.
4. You could repeat Step 1 for multiple times (but less than 5 times), or directly skip Step 1 if user is clear initially.
--- Step 2: summarize user's intention ---
1. Make the summary as long as information is enough. You do not need every details to do summary.
2. Make the summary comprehensive, short, and precise, based on your interactions with user.

Think step by step, first give a thought [THOUGHT] and then give your inquiry [INQUIRY] or summary [SUMMARY]."""

def initialize():
    # get arguments
    parser = argparse.ArgumentParser("")
    # model arguments
    parser.add_argument("--model_name_or_path", default='/data/qiancheng/ckpts/mistral-7b')
    parser.add_argument("--load_ckpt", type=str, default="/data/qiancheng/ckpts/step_1161_M_w_S")
    parser.add_argument("--data_dir", type=str, default='/data/qiancheng/datasets/interaction_data_test.jsonl')
    parser.add_argument('--output_dir', type=str, default='/data/qiancheng/output')

    args = parser.parse_args()
    # init bmt 
    bmt.init_distributed(seed=0, zero_level=3)

    return args

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

def setup_model_and_tokenizer(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args)
    return tokenizer, model


def preprocess_data(task, actions):
    test_prompt = f"<s>User: {SYS_PROMPT}\n\nHere is the task:\n{task}\nAgent: "
    return {"task": task, "prompt": test_prompt, "label": actions}

def load_raw_dataset(args):
    tasks = []
    with open(args.data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            task = json.loads(line)["task"]
            actions = json.loads(line)["actions"]
            processed_task = preprocess_data(task, actions)
            tasks.append(processed_task)
    return tasks

def main_loop(args, test_dataset, tokenizer, model):
    BeamGen = LlamaRandomSampling(model, tokenizer)
    for i, data in enumerate(test_dataset):
        prompt = data["prompt"]
        ori_prompt = copy.deepcopy(prompt)
        task = data["task"]
        print("~~~~~~~~~~~~~~~~~ Begin New Task ~~~~~~~~~~~~~~~~~")
        print(f"User Task: {task}")
        actions = [{"role": "User", "thought": None, "content": task, "type": "response"}]
        back_flag = False
        while True:
            if not back_flag:
                print("=-=-=-=-= Generating ... =-=-=-=-=")
                preds = BeamGen.generate(
                    [prompt],
                    max_length=1024,
                    repetition_penalty=1.2,
                    # temperature=0.9,
                    # top_p=0.95,
                    # top_k=40,
                )
                pred = preds[0]
                if "[SUMMARY]" in pred:
                    thought = pred.split("[THOUGHT]")[1].split("[SUMMARY]")[0].strip()
                    response = pred.split("[SUMMARY]")[1].strip()
                    print(f"Assistent Thought: {thought}\nAssistant Summary: {response}")
                    actions.append({"role": "Assistant", "thought": thought, "content": response, "type": "summary"})
                    break
                thought = pred.split("[THOUGHT]")[1].split("[INQUIRY]")[0].strip()
                response = pred.split("[INQUIRY]")[1].strip()
                print(f"Assistent Thought: {thought}\nAssistant Response: {response}")
                actions.append({"role": "Assistant", "thought": thought, "content": response, "type": "response"})
                prompt += pred + "\n"
            else:
                back_flag = False
            
            # Get in time user response
            user_input = input("Your Response: ")
            if user_input.strip() == "back":
                back_flag = True
                actions = actions[:-2]
                prompt = "User: ".join(prompt.split("User: ")[:-1]).strip() + "\n"
                print("=-=-=-=-= Rolling back to ... =-=-=-=-=")
                print(prompt.split("Here is the task:")[1].strip())
                print("=-=-=-=-=    =-=-=-=-=-=-=    =-=-=-=-=")
            elif user_input.strip() != "":
                prompt += "User: " + user_input + "\nAgent: "
                actions.append({"role": "User", "thought": None, "content": user_input, "type": "response"})
            else:
                break
        # save test results
        save_path = os.path.join(args.output_dir, args.load_ckpt.split("/")[-1])
        os.makedirs(save_path, exist_ok=True)
        logger.info(f" Saving results of test {i} ...")
        with open(os.path.join(save_path, f'test_{str(i)}.json'), "w", encoding='utf-8') as fout:
            fout.write(json.dumps({"prompt": ori_prompt, "actions": actions}, ensure_ascii=False, indent=4))

def main():
    args = initialize()
    tokenizer, model = setup_model_and_tokenizer(args)
    raw_tasks = load_raw_dataset(args)
    main_loop(args, raw_tasks, tokenizer, model)

if __name__ == "__main__":
    main()