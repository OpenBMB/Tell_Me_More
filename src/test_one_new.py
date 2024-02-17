import argparse
import bmtrain as bmt
import os
import random
import copy
import json
from cprint import cprint
from IPython import embed
from model_center.model import Llama
from model_center.tokenizer import LlamaTokenizer
from model_center.generation.llama import LlamaRandomSampling
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# SYS_PROMPT = """You are trying to understand the user's intention. The user will provide a task which maybe vague or clear (they may not provide their own preferences).
# 1. In the very first round of thought, you should explicitly judge if the task is vague or clear and why.
# 2. If the task is vague, you should ask the user for more information with options for user to choose from. If it is clear, then do not query and repeat the user's task in the summary.
# 3. Please only ask one question with options at a time. You could ask the user for multiple rounds until you think the user's goal is clear enough.
# 4. Your questions should be about different missing details and aspects, be diverse.
# 5. When you think you have gathered enough information, you should provide a summary of the detailed user's detailed goal
# 6. Do not solve the task, just provide a detailed summary about the task and user preference in the end. Summarize the information you got in 1-2 sentences.

# You should follow the format, first provide your thought, then ask the user for more information or provide a summary: 
# Thought: ... Query: ... (You would like to query the user)
# OR: Thought: ... Summary: ... (You have gathered enough information and make summary)
# The user will provide a task as the following."""

SYS_PROMPT = """You are an agent trying to understand user's goal and summarize it. Please first ask users for more specific details with options, and finally summarize the user's intention. Here are steps in detail and corresponding EXAMPLEs.

--- Step 1: initial thought generation ---
1. Generate [INITIAL THOUGHT] about if the task is vague or clear and why.
2. List the important missing details and some according options if the task is vague.
### EXAMPLE:
[INITIAL THOUGHT] The task is to research methods to secure a home WiFi network. This is a broad topic and could include a variety of methods such as changing default passwords, using WPA3 encryption, enabling a firewall, etc. However, the user has not specified any particular aspect of security they are interested in, the level of technical expertise they have, or if they are looking for basic or advanced security measures. These details would help narrow down the research to provide more targeted information. Some aspects of missiong details and potential options are as follows:
- Specific aspect of security: General best practices, Specific threats like hacking or malware, Parental controls and content filtering
- User's technical expertise: Beginner, Intermediate, Advanced
- Preference for security level: Basic security measures, Advanced security options

--- Step 2: inquiry for more information if vague ---
1. If the task is vague, inquiry more details with options according to the list in [INITIAL THOUGHT].
2. Think what information you have and what to inquiry next in [INQUIRY THOUGHT].
3. Present your inquiry with options for user to choose after [INQUIRY], and be friendly.
4. You could repeat Step 2 for multiple times (but less than 5 times), or directly skip Step 2 if user task is clear initially.
### EXAMPLE:
[INQUIRY THOUGHT] I need to understand which aspect of security the user wants to learn about. Starting with this will help tailor the information to their specific needs.
[INQUIRY] Sure, I can help you with that! Are you looking for general best practices to secure your WiFi network, or are you specifically concerned about certain threats like hacking or malware? Maybe you're interested in parental controls and content filtering?

--- Step 3: summarize user's intention ---
1. Make the summary once information is enough. You do not need to inquiry every missing details in [INITIAL THOUGHT].
2. List all the user's preferences and constraints in [SUMMARY THOUGHT]. The number of points should be same as rounds of chatting.
3. Give final summary after [SUMMARY] with comprehensive details in one or two sentences.
### EXAMPLE:
[SUMMARY THOUGHT] The user has provided specific constraints over the course of three interactions which now allow for a clear understanding of their intention. Here are the user preferences and constraints:
- Focus on general best practices to secure WiFi network.
- User is technically advanced.
- Looking for basic security measures only.
[SUMMARY] The user seeks information on basic, general best practices for securing a home WiFi network suitable for someone with an advanced level of technical expertise."""

def initialize():
    # get arguments
    parser = argparse.ArgumentParser("")
    # model arguments
    parser.add_argument("--model_name", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str, default='')
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--start_from', type=int, default=0)

    args = parser.parse_args()
    # init bmt 
    bmt.init_distributed(seed=0, zero_level=3)

    return args

def get_multiple_lines_input(msg: str):
    lines = []
    i = 0
    while True:
        if i == 0:
            user_input = input(msg).strip()
        else:
            user_input = input().strip()
        i += 1
        # 👇️ if user pressed Enter without a value, break out of loop
        if user_input == '':
            break
        else:
            lines.append(user_input + '\n')
    # 👇️ join list into a string
    return ''.join(lines).strip()

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

def preprocess_data(task):
    test_prompt = f"<s>User: {SYS_PROMPT}\n\nHere is the task:\n{task}\nAgent: "
    return {"task": task, "prompt": test_prompt}

def load_raw_dataset(args, tokenizer):
    tasks = []
    with open(args.data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            task = json.loads(line)["task"]
            if args.model_name == "mistral-interact" or args.model_name == "llama2-interact":
                processed_task = preprocess_data(task)
            else:
                raise ValueError(f"Not supported model {args.model_name}")
            tasks.append(processed_task)
    
    # random.seed(233)
    # random.shuffle(tasks)
    return tasks[args.start_from:]

def main_loop(args, test_dataset, tokenizer, model):
    BeamGen = LlamaRandomSampling(model, tokenizer)
    i = -1
    # task = input("Give Your New Task: ").strip()
    for i, data in enumerate(test_dataset[:]):
    # while task != "":
        # i += 1
        prompt = data["prompt"]
        # prompt = f"<s>User: {SYS_PROMPT}\n\nHere is the task:\n{task}\nAgent: "
        ori_prompt = copy.deepcopy(prompt)
        task = data["task"]
        save_dict = {}
        cprint.err("~~~~~~~~~~~~~~~~~ Begin New Task ~~~~~~~~~~~~~~~~~")
        cprint.warn(f"User Task: {task}")
        actions = [{"role": "User", "thought": None, "content": task, "type": "response"}]
        back_flag = False
        summary_flag = False
        vague = True
        
        while True:
            if not back_flag:
                print("=-=-=-=-= Generating ... =-=-=-=-=")
                round = 0
                while round < 3:
                    round += 1
                    try:
                        preds = BeamGen.generate(
                            [prompt],
                            max_length=1024,
                            repetition_penalty=1.2,
                            temperature=0.2,
                            top_p=0.95,
                            # top_k=40,
                        )
                        pred = preds[0]
                        if args.model_name == "mistral-interact" or args.model_name == "llama2-interact":
                            # Parse Initial Thought
                            if "[INITIAL THOUGHT]" in pred:
                                initial_thought:str = pred.split("[INITIAL THOUGHT]")[1].split("[INQUIRY THOUGHT]")[0].strip()
                                cprint.ok(f"Initial Thought: {initial_thought}")
                                save_dict["initial_thought"] = initial_thought
                                if "[SUMMARY]" in pred:
                                    vague = False
                                    save_dict["user_record"] = {"missing_details_num": 0, "missing_with_options": 0, "total_options": 0, "inappropriate_options": 0, "inappropriate_options_reason": None}
                                else:
                                    missing_num = initial_thought.count("\n-") # int(input("What's the number of missing details? ").strip())
                                    missing_with_op = min(initial_thought.count(": "), missing_num) # int(input("What's the total number of missing details with options? ").strip())
                                    total_options = '\n-'.join(initial_thought.split("\n-")[1:]).count(",") + len(initial_thought.split("\n-")[1:])
                                    cprint.warn(f"Auto Parsing:\nThe number of missing details is {str(missing_num)}\nThe number missing details with options is {str(missing_with_op)}\nThe total number of options is {str(total_options)}\nPress Enter if it's correct ...")
                                    user_input = input()
                                    if user_input.strip() != "":
                                        missing_num = int(input("What's the number of missing details? ").strip())
                                        missing_with_op = int(input("What's the total number of missing details with options? ").strip())
                                        total_options = int(input("What's the total number of options? ").strip())
                                    # total_options = int(input("What's the total number of options? ").strip())
                                    inappropriate_options = int(input("What's the number of options you think is unreasonable? ").strip())
                                    inappropriate_options_reasons = None
                                    if inappropriate_options != 0:
                                        inappropriate_options_reasons = int(input("Give some reasons why: ").strip())
                                    save_dict["user_record"] = {"missing_details_num": missing_num, "missing_with_options": missing_with_op, "total_options": total_options, "inappropriate_options": inappropriate_options, "inappropriate_options_reason": inappropriate_options_reasons}
                            
                            # Parse the summary
                            if "[SUMMARY]" in pred:
                                thought = pred.split("[SUMMARY THOUGHT]")[1].split("[SUMMARY]")[0].strip()
                                response = pred.split("[SUMMARY]")[1].strip()
                                summary_flag = True
                            else:
                                thought = pred.split("[INQUIRY THOUGHT]")[1].split("[INQUIRY]")[0].strip()
                                response = pred.split("[INQUIRY]")[1].strip()
                        
                        break
                    except Exception as e:
                        print(pred)
                        cprint.fatal(f"Parsing Error: {e}\n=-=-=-=-= Re-Generating time: {round}... =-=-=-=-=")
                
                # Process the information got from the prediction
                if summary_flag:
                    cprint.ok(f"Assistant Thought: {thought}\nAssistant Summary: {response}")
                    total_user_details = int(input("What's the total number of detailed information you provided? ").strip())
                    user_details_in_summary = int(input("What's the number of details that is explicitly summarized in the summary? ").strip())
                    if "user_record" not in save_dict:
                        if not vague:
                            save_dict["user_record"] = {"missing_details_num": 0, "missing_with_options": 0, "total_options": 0, "inappropriate_options": 0, "inappropriate_options_reason": None}
                        else:
                            raise Exception("No initial thought generated!")
                    save_dict["user_record"]["total_user_details"] = total_user_details
                    save_dict["user_record"]["user_details_in_summary"] = user_details_in_summary
                    save_dict["vague"] = vague
                    actions.append({"role": "Assistant", "thought": thought, "content": response, "type": "summary"})
                    prompt += pred
                else:
                    cprint.info(f"Assistant Thought: {thought}\nAssistant Response: {response}")
                    option_num = int(input("How many options are given in the 'Assistant Response' provided? ").strip())
                    inappropriate_option_num = int(input("How many options given do you think is unreasonable? ").strip())
                    actions.append({"role": "Assistant", "thought": thought, "content": response, "type": "response", "option_num": option_num, "inappropriate_option_num": inappropriate_option_num})
                    if args.model_name == "mistral-interact" or args.model_name == "llama2-interact":
                        prompt += pred + "\n"
                    else:
                        prompt += pred + " </s>"
            else:
                back_flag = False
                summary_flag = False

            # Get in time user response
            if not summary_flag:
                user_input = input("Your Response: ")
            else:
                user_input = input("Type `Enter` to save, or `back` to modify: ")

            # Process the user input
            if user_input.lower().strip() == "back":
                back_flag = True
                actions = actions[:-2]
                if args.model_name == "mistral-interact" or args.model_name == "llama2-interact":
                    prompt = "User: ".join(prompt.split("User: ")[:-1]).strip() + "\n"
                else:
                    prompt = "<s>[INST] ".join(prompt.split("<s>[INST] ")[:-1]).strip()
                print("=-=-=-=-= Rolling back to ... =-=-=-=-=")
                print(prompt.split("Here is the task:")[1].strip())
                print("=-=-=-=-=    =-=-=-=-=-=-=    =-=-=-=-=")
            elif user_input.strip() != "":
                if args.model_name == "mistral-interact" or args.model_name == "llama2-interact":
                    prompt += "User: " + user_input + "\nAgent: "
                else:
                    prompt += "<s>[INST] " + user_input + " [/INST]\n"
                actions.append({"role": "User", "thought": None, "content": user_input, "type": "response"})
            else:
                break
        
        # save test results
        save_dict["prompt"] = prompt
        save_dict["actions"] = actions
        save_path = args.output_dir # os.path.join(args.output_dir, args.load_ckpt.split("/")[-1])
        os.makedirs(save_path, exist_ok=True)
        logger.info(f" Saving results of test {args.start_from + i} ...")
        with open(os.path.join(save_path, f'user_interaction_record_{args.model_name}.jsonl'), "a", encoding='utf-8') as fout:
            fout.write(json.dumps(save_dict) + "\n")
        # task = input("Give Your New Task: ")

def main():
    args = initialize()
    tokenizer, model = setup_model_and_tokenizer(args)
    raw_tasks = load_raw_dataset(args, tokenizer)
    main_loop(args, raw_tasks, tokenizer, model)

if __name__ == "__main__":
    main()