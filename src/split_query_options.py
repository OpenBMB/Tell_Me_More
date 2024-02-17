import os
import copy
import json
from tqdm import tqdm
from IPython import embed
from utils import gpt_chatcompletion

# TODO: "gpt4 / Llama-2-7b-chat-hf / mistral-7b-instruct-v0.2-hf / mistral-interact"
model_name = "gpt4" 
interaction_data_path = f'./data/user_interaction_records/user_interaction_record_{model_name}.jsonl'
interaction_dataset = []

with open(interaction_data_path, 'r', encoding='utf-8') as f:
    for line in f:
        interaction_dataset.append(json.loads(line))

"""
0. Split the inquiry from Assistant into separate query and options
"""
SYS_PROMPT = """You are a helpful assistant in understanding and analyzing user's queries. You are tasked with two missions:
1. What queries does the user make, disentangle them and list them line by line. You should identify the specific querying aspects (instead of general ones) and disentangle them.
2. For each query, what options does the user provide, list them after the query and "--" one by one, separated by a comma. If no options are parsed, leave the space after "--" empty.
Each time, the user will present a query. Please respond following the format given in the chat history. First, provide your thought, then present the response in a structured format."""

BASE_MESSAGES = [
    {'role': 'system', 'content': SYS_PROMPT},
    {'role': 'user', 'content': "Can you tell me a bit more about the novel ABC? What is the main plot or theme of the novel?"},
    {'role': 'assistant', 'content': """Thought:
The user is asking for two specific aspects of the novel ABC but does not provide the option for any query.
Response:
What's the main plot of the novel ABC? --
What's the theme of the novel ABC? --"""},
    {'role': 'user', 'content': "Awesome, there are usually some cool ones in China! How much are you looking to spend on attending the festival? Would it be under $500, between $500 - $1500, or above $1500?"},
    {'role': 'assistant', 'content': """Thought:
The user only asks one query with three options.
Response:
How much are you looking to spend on attending the festival? -- under $500, between $500 - $1500, above $1500"""},
    {'role': 'user', 'content': "Can you please provide more details about the music festival you're looking for? For example, what type of music do you prefer (e.g. rock, pop, electronic)? Is it a local festival or one that's happening in a different city? Are there any specific dates or venues you're interested in?"},
    {'role': 'assistant', 'content': """Thought:
The user asks for the type of music, where it is held, dates, and venues. There are in total 4 aspects that should be disentangled.
Response:
What type of music do you prefer? -- rock, pop, electronic
Where do you prefer the music festival to be located? -- locally, in a different city
What's the date of the music festival? --
What's the venue of the music festival? --"""},
]

TASK_NUM = len(interaction_dataset)

for i in range(TASK_NUM):
    print(f"=========== Processing task {i} ===========")
    assert len(interaction_dataset[i]['actions']) % 2 == 0
    turn_num = len(interaction_dataset[i]['actions']) // 2 - 1
    task_query_options_list = []
    for j in tqdm(range(turn_num)):
        messages = copy.deepcopy(BASE_MESSAGES)
        messages.append({
            "role": "user",
            "content": interaction_dataset[i]['actions'][2 * j + 1]['content']
        })
        resp = gpt_chatcompletion(messages)
        resp = resp.split("Response:")[-1].strip().split('\n')
        query_options_list = []
        for data in resp:
            try:
                query, options = data.split('--')
            except Exception as e:
                print("Exception: ", e)
                print(data)
            query_options_list.append({
                "query": query.strip(),
                "options": [option.strip() for option in options.split(',')] if options.strip() != '' else [],
            })
        task_query_options_list.append(query_options_list)
    interaction_dataset[i]['query_options_list'] = task_query_options_list

with open(f'./data/user_interaction_records/user_interaction_record_{model_name}_split.jsonl', 'w', encoding='utf-8') as f:
    for data in interaction_dataset:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')