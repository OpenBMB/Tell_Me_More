import os
import copy
import json
from tqdm import tqdm
from IPython import embed
from utils import gpt_chatcompletion


def form_messages(system_prompt: str, msg: str):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': msg}
    ]
    return messages


def tell_vague(data):
    return data['vague']


label_setting = 'mix'
# TODO: "gpt4 / Llama-2-7b-chat-hf / mistral-7b-instruct-v0.2-hf / mistral-interact"
model_name = "mistral-interact" 
interaction_data_path = f'./data/user_interaction_records/user_interaction_record_{model_name}_split.jsonl'
labeller_data_path = f"./data/data_labeling/test_data_report_{label_setting}.jsonl"
metric_save_path = f"./data/user_interaction_records/metrics/metric_{label_setting}_{model_name}.json"

interaction_dataset = []
labeller_dataset = []

with open(interaction_data_path, 'r', encoding='utf-8') as f:
    for line in f:
        interaction_dataset.append(json.loads(line))

with open(labeller_data_path, 'r', encoding='utf-8') as f:
    for line in f:
        labeller_dataset.append(json.loads(line))

QUERY_GPT = True
TASK_NUM = len(interaction_dataset)
results = {}

"""
1. vagueness_judgement_accuracy: 判断 vague/clear 与 human intention 的一致率
"""
align_cnt = 0
for i in range(TASK_NUM):
    if tell_vague(interaction_dataset[i]) == labeller_dataset[i]['user_vague']:
        align_cnt += 1
vagueness_judgement_accuracy = align_cnt / TASK_NUM
results['vagueness_judgement_accuracy'] = vagueness_judgement_accuracy


"""
2. missing_details_recover_rate: 模型给出的 missing details 与 human intention 的一致率（按重要程度划分）
"""

SYS_PROMPT = """You are a helpful assistant and good at judging similarities between phrases. Given a QUESTION and a list of phrases, you should determine if the provided QUESTION semantically matches any of the entries in the list. Directly answer with the phrase in the list if there is any semantic match, and 'None of the above' otherwise. Do not include any explanations and additional information. Remember to strictly follow the format given in the following examples:

1. Example with a semantic match:

### QUESTION:
What is the time frame for the price predictions?

### List of phrases:
- Historical data timeframe and granularity
- Criteria for efficiency
- Specific stocks or market sectors
- Computational resources available
- Type of historical data

### Answer:
Historical data timeframe and granularity

2. Example without a semantic match:

### QUESTION:
What is the time frame for the price predictions?

### List of phrases:
- Metrics or sources to determine popularity
- User's personal style preferences
- Geographic region
- Specific fashion categories

### Answer:
None of the above"""

USER_PROMPT = """Here is a QUESTION and a list of phrases:

### QUESTION:
{question}

### List of phrases:
{list_str}"""

vague_task_cnt = 0
missing_details_recover_rate = {
    "1": {"rate": 0.0, "cnt": 0},
    "2": {"rate": 0.0, "cnt": 0},
    "3": {"rate": 0.0, "cnt": 0},
    "total_recover_rate": {"rate": 0.0, "cnt": 0},
}
for i in range(TASK_NUM):
    print(f"=========== Processing task {i} ===========")
    if tell_vague(interaction_dataset[i]) and labeller_dataset[i]['user_vague']:
        vague_task_cnt += 1
        
        human_intention_info = []
        human_intention_info.extend(labeller_dataset[i]['user_approve'])
        human_intention_info.extend(labeller_dataset[i]['user_rectify'])
        human_intention_info.extend(labeller_dataset[i]['user_add'])
        
        list_str = [f"- {info['description']}" for info in human_intention_info]
        list_str = '\n'.join(list_str)
        
        flag_dict = {info['description'].lower(): {"hit": False, "importance": info['importance']} for info in human_intention_info}
        
        for turn_info in interaction_dataset[i]['query_options_list']:
            for query_options in turn_info:
                question = query_options['query']
                msg = USER_PROMPT.format(question=question, list_str=list_str)
                resp = gpt_chatcompletion(form_messages(SYS_PROMPT, msg)) if QUERY_GPT else "yes"
                if resp != "None of the above":
                    resp = resp.lower()
                    if resp in flag_dict:
                        flag_dict[resp]['hit'] = True
                    else:
                        print(f"Error: {resp} not in human intention list:\n{list_str}")

        task_i_results = {}
        for k, v in flag_dict.items():
            if v['importance'] not in task_i_results:
                task_i_results[v['importance']] = {'hit': 0, 'total': 0}
            task_i_results[v['importance']]['total'] += 1
            if v['hit']:
                task_i_results[v['importance']]['hit'] += 1
        
        for importance in task_i_results:
            missing_details_recover_rate[importance]['rate'] += task_i_results[importance]['hit'] / task_i_results[importance]['total']
            missing_details_recover_rate[importance]['cnt'] += 1
        missing_details_recover_rate['total_recover_rate']['rate'] += sum([v['hit'] for v in flag_dict.values()]) / len(flag_dict)
        missing_details_recover_rate['total_recover_rate']['cnt'] += 1
        
        print(f"task_i_results: {task_i_results}")
        print(f"missing_details_recover_rate: {missing_details_recover_rate}")
        
for importance in missing_details_recover_rate:
    missing_details_recover_rate[importance]['cnt'] = max(missing_details_recover_rate[importance]['cnt'], 1)
    missing_details_recover_rate[importance] = missing_details_recover_rate[importance]['rate'] / missing_details_recover_rate[importance]['cnt']
results['missing_details_recover_rate'] = missing_details_recover_rate


"""
3. summary_intention_coverage_rate: 模型给出的 summary 对 User Response 的覆盖率
"""
coverage = 0.0
vague_task_cnt = 0
for i in range(TASK_NUM):
    if tell_vague(interaction_dataset[i]):
        vague_task_cnt += 1
        user_record = interaction_dataset[i]['user_record']
        user_record['total_user_details'] = max(user_record['total_user_details'], 1)
        cov_i = user_record['user_details_in_summary'] / user_record['total_user_details']
        coverage += cov_i
summary_intention_coverage_rate = coverage / vague_task_cnt
results['summary_intention_coverage_rate'] = summary_intention_coverage_rate


"""
4. options_presenting_rate: 模型给出的 missing details 含有 options 的比率
"""
vague_task_cnt = 0
options_presenting_rate = 0.0
for i in range(TASK_NUM):
    if tell_vague(interaction_dataset[i]):
        vague_task_cnt += 1
        task_i_num_missing_details = 0
        task_i_num_missing_details_with_options = 0
        for turn_info in interaction_dataset[i]['query_options_list']:
            for query_options in turn_info:
                task_i_num_missing_details += 1
                if len(query_options['options']) > 0:
                    task_i_num_missing_details_with_options += 1
        options_presenting_rate += task_i_num_missing_details_with_options / task_i_num_missing_details
options_presenting_rate /= vague_task_cnt
results['options_presenting_rate'] = options_presenting_rate


"""
5. options_reasonable_rate: 模型与用户的 conversation 中 options 的合理程度
"""
vague_task_cnt = 0
options_reasonable_rate = 0.0
for i in range(TASK_NUM):
    if tell_vague(interaction_dataset[i]):
        vague_task_cnt += 1
        assert len(interaction_dataset[i]['actions']) % 2 == 0
        turn_num = len(interaction_dataset[i]['actions']) // 2 - 1

        total_options_interact = 0
        total_inappropriate_options_interact = 0
        for j in range(turn_num):
            total_options_interact += interaction_dataset[i]['actions'][2 * j + 1]['option_num']
            total_inappropriate_options_interact += interaction_dataset[i]['actions'][2 * j + 1]['inappropriate_option_num']
        total_options_interact = max(total_options_interact, 1)
        options_reasonable_rate += 1 - total_inappropriate_options_interact / total_options_interact

options_reasonable_rate /= vague_task_cnt
results['options_reasonable_rate'] = options_reasonable_rate


"""
6. average_provided_options: 模型给出的 missing details 含有平均多少个 options
"""
vague_task_cnt = 0
average_provided_options = 0.0
for i in range(TASK_NUM):
    if tell_vague(interaction_dataset[i]):
        vague_task_cnt += 1
        task_i_num_missing_details = 0
        task_i_num_options = 0
        for turn_info in interaction_dataset[i]['query_options_list']:
            for query_options in turn_info:
                task_i_num_missing_details += 1
                task_i_num_options += len(query_options['options'])
        average_provided_options += task_i_num_options / task_i_num_missing_details
average_provided_options /= vague_task_cnt
results['average_provided_options'] = average_provided_options


"""
7. average_inquired_missing_details_per_round: 模型每一轮给出的问题的平均数目
"""
vague_task_cnt = 0
average_inquired_missing_details_per_round = 0.0
for i in range(TASK_NUM):
    if tell_vague(interaction_dataset[i]):
        vague_task_cnt += 1
        task_i_num_missing_details = 0
        num_turn = len(interaction_dataset[i]['query_options_list'])
        for turn_info in interaction_dataset[i]['query_options_list']:
            task_i_num_missing_details += len(turn_info)
        average_inquired_missing_details_per_round += task_i_num_missing_details / num_turn
average_inquired_missing_details_per_round /= vague_task_cnt
results['average_inquired_missing_details_per_round'] = average_inquired_missing_details_per_round


"""
8. average_conversation_rounds: 平均对话轮次
"""
average_conversation_rounds = 0.0
for i in range(TASK_NUM):
    assert len(interaction_dataset[i]['actions']) % 2 == 0
    average_conversation_rounds += len(interaction_dataset[i]['actions']) // 2
average_conversation_rounds /= TASK_NUM
results['average_conversation_rounds'] = average_conversation_rounds


"""
9. average_inquired_missing_details: 模型给出的问题的平均数目
"""
vague_task_cnt = 0
average_inquired_missing_details = 0.0
for i in range(TASK_NUM):
    if tell_vague(interaction_dataset[i]):
        vague_task_cnt += 1
        task_i_num_missing_details = 0
        for turn_info in interaction_dataset[i]['query_options_list']:
            task_i_num_missing_details += len(turn_info)
        average_inquired_missing_details += task_i_num_missing_details
average_inquired_missing_details /= vague_task_cnt
results['average_inquired_missing_details'] = average_inquired_missing_details


os.makedirs(metric_save_path, exist_ok=True)
with open(metric_save_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(results, indent=4))