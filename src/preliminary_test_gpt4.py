from IPython import embed
import json
import time
from cprint import cprint
import os
from copy import deepcopy
from utils import gpt_chatcompletion

SYS_PROMPT = """You are trying to understand the user's intention. The user will provide a task which may be vague or clear (they may not provide their own preferences).
1. In the very first round of thought, you should explicitly judge if the task is vague or clear and why, also you should list the important missing details and some according options if the task is vague.
2. If the task is vague, you should ask the user for more information with options for user to choose from. If it is clear, then do not query and repeat the user's task in the summary.
3. Please only ask one question with options at a time. You could ask the user for multiple rounds until you think the user's goal is clear enough.
4. Your questions should be about different missing details and aspects, be diverse.
5. When you think you have gathered enough information, you should provide a summary of the detailed user's detailed goal
6. Do not solve the task, just provide a detailed summary about the task and user preference in the end. Summarize the information you got in 1-2 sentences.

You should follow the format, first provide your thought, then ask the user for more information or provide a summary. Here are four examples of responses, the format of which you need to follow:

### EXAMPLE-1: If the task is clear, in the very first round, give your [INITIAL THOUGHT], followed by [SUMMARY THOUGHT] and [SUMMARY].
[INITIAL THOUGHT] The task is clear. The user has specified the currencies of interest (USD to Euro) and is asking for the current exchange rate. No additional details or preferences are needed to fulfill this task.
[SUMMARY THOUGHT] The initial task is straightforward as it involves a specific request to obtain a piece of information that does not require further clarification or personal preference.
[SUMMARY] The user's goal is to find out the current exchange rate from USD to Euro, with no additional constraints or preferences mentioned.

### EXAMPLE-2: If the task is vague, in the very first round, give your [INITIAL THOUGHT], followed by [INQUIRY THOUGHT] and [INQUIRY].
[INITIAL THOUGHT] The task is vague because it does not specify the user's current financial situation, income, expenses, spending habits, or specific areas they are willing to cut back on. Without this information, it's difficult to provide tailored advice on saving money. Some aspects of missiong details and potential options are as follows:
- Current financial situation: precise monthly income and expenses,  detailed spending breakdown
- Willingness to change lifestyle: Reduce dining out, Cancel subscriptions, Delay major purchases
- Financial goals: Emergency fund, Savings for a purchase, Investment
[INQUIRY THOUGHT] I need to understand the user's current financial situation to offer relevant advice on saving money.
[INQUIRY] To help you save money, could you tell me a bit about your monthly income and expenses? Knowing where your money goes can make it easier to find areas where we might be able to trim down.

### EXAMPLE-3: If the task is vague, after the first round of dialogue, give your [INQUIRY THOUGHT] and [INQUIRY] about the missing details in [INITIAL THOUGHT] if the information is not enough.
[INQUIRY THOUGHT] The user has provided their income range, but we still need to know about their expenses to give advice on saving.
[INQUIRY] Thanks for sharing that! Could you also let me know about your monthly expenses? Are they generally higher or lower than your income, and do you have any specific areas where you spend a lot, like housing, transportation, or entertainment?

### EXAMPLE-4: If the task is vague, and the information is enough, directly give your [SUMMARY THOUGHT] and [SUMMARY].
[SUMMARY THOUGHT] The user has provided sufficient information over the course of our conversation to understand their financial situation and preferences, which allows for a tailored response. Here are the user preferences and constraints:
- Monthly income is between $10000 and $12000
- Monthly expenses are about $5000, mainly on games and foods
- No willingness to reduce gaming or dining out
- Expenses include book purchases; open to alternative methods to save without affecting reading habit
[SUMMARY] The user earns $10000-$12000 monthly with expenses around $5000, focusing on gaming and food. They want to save money but aren't ready to cut back on these areas. To save on books, they prefer alternatives that don't impact their reading enjoyment, such as digital versions or library loans.

Remember you must strictly conform to one of the above example formats, first provide your thought, then ask the user for more information or provide a summary. The user will provide a task as the following."""

model_name = 'gpt-4'
data_dir = './vagueness_augmented_test.jsonl'
output_dir = './interaction_output'

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

def preprocess_gpt_data(task):
    "For gpt-4"
    messages = [
        {"role": "system", "content": f"{SYS_PROMPT}"},
        {"role": "user", "content": f"Here is my task: {task}"},
    ]
    return {"task": task, "messages": messages}

def load_raw_dataset():
    tasks = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            task = json.loads(line)["task"]
            processed_task = preprocess_gpt_data(task)
            tasks.append(processed_task)
    return tasks

test_dataset = load_raw_dataset()

i = -1
for i, data in enumerate(test_dataset[:]):

    messages = data["messages"]
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
                    pred = gpt_chatcompletion(messages)

                    cprint.ok(f"Prediction:\n{pred}")
                    has_initial_thought = input("Is there an initial thought in prediction? (y or n) ").strip()
                    has_summary = input("Is there an summary in prediction? (y or n) ").strip()

                    if has_initial_thought == 'y':
                        initial_thought = get_multiple_lines_input("Copy the initial thought here: ")
                        # initial_thought: str = input("Copy the initial thought here: ").strip()
                        save_dict["initial_thought"] = initial_thought

                        if has_summary == 'y':
                            vague = False
                            save_dict["user_record"] = {"missing_details_num": 0, "missing_with_options": 0, "total_options": 0, "inappropriate_options": 0, "inappropriate_options_reason": None}
                        else:
                            missing_num = int(input("What's the number of missing details in initial thought? ").strip())
                            missing_with_op = int(input("What's the total number of missing details with options in initial thought? ").strip())
                            total_options = int(input("What's the total number of options in initial thought? ").strip())
                            inappropriate_options = int(input("What's the number of options you think is unreasonable in initial thought? ").strip())
                            inappropriate_options_reasons = None
                            if inappropriate_options != 0:
                                inappropriate_options_reasons = int(input("Give some reasons why: ").strip())
                            save_dict["user_record"] = {"missing_details_num": missing_num, "missing_with_options": missing_with_op, "total_options": total_options, "inappropriate_options": inappropriate_options, "inappropriate_options_reason": inappropriate_options_reasons}

                    if has_summary == 'y':
                        thought = get_multiple_lines_input("Copy the summary thought here: ")
                        response = get_multiple_lines_input("Copy the summary here: ")
                        # thought = input("Copy the summary thought here: ").strip()
                        # response = input("Copy the summary here: ").strip()
                        summary_flag = True
                    else:
                        thought = get_multiple_lines_input("Copy the inquiry thought here: ")
                        response = get_multiple_lines_input("Copy the inquiry here: ")
                        # thought = input("Copy the inquiry thought here: ").strip()
                        # response = input("Copy the inquiry here: ").strip()
                        
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
            else:
                cprint.info(f"Assistant Thought: {thought}\nAssistant Response: {response}")
                option_num = int(input("How many options are given in the 'Assistant Response' provided? ").strip())
                inappropriate_option_num = int(input("How many options given do you think is unreasonable? ").strip())
                actions.append({"role": "Assistant", "thought": thought, "content": response, "type": "response", "option_num": option_num, "inappropriate_option_num": inappropriate_option_num})
            messages.append({"role": "assistant", "content": response})
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
            messages = messages[:-2]
            print("=-=-=-=-= Rolling back to ... =-=-=-=-=")
            print("=-=-=-=-=    =-=-=-=-=-=-=    =-=-=-=-=")
        elif user_input.strip() != "":
            messages.append({"role": "user", "content": user_input})
            actions.append({"role": "User", "thought": None, "content": user_input, "type": "response"})
        else:
            break
    
    # prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    # # save test results
    # save_dict["prompt"] = prompt
    save_dict["actions"] = actions
    save_path = output_dir # os.path.join(args.output_dir, args.load_ckpt.split("/")[-1])
    os.makedirs(save_path, exist_ok=True)
    print(f" Saving results of test {i} ...")
    with open(os.path.join(save_path, f'user_interaction_record_{model_name}.jsonl'), "a", encoding='utf-8') as fout:
        fout.write(json.dumps(save_dict) + "\n")
    # task = input("Give Your New Task: ")