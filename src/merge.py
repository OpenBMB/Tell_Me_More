import json

# TODO: "mix / hbx+qc / dengjia"
label_setting = 'mix'

# TODO: "gpt4 / Llama-2-7b-chat-hf / mistral-7b-instruct-v0.2-hf / mistral-interact"
model_name_list = [
    "mistral-7b-instruct-v0.2-hf",
    "Llama-2-7b-chat-hf",
    "gpt4",
    "mistral-interact"
]

merged_metric = {}

for model_name in model_name_list:
    metric_save_path = f"../data/user_interaction_records/metrics/metric_{label_setting}_{model_name}.json"
    with open(metric_save_path, 'r', encoding='utf-8') as f:
        metric = json.load(f)
    for k, v in metric.items():
        if k not in merged_metric:
            merged_metric[k] = {}
        merged_metric[k][model_name] = v
        if isinstance(v, dict):
            for kk, vv in v.items():
                merged_metric[k][model_name][kk] = round(vv, 2)
        else:
            merged_metric[k][model_name] = round(v, 2)

merged_metric_save_path = f"../data/user_interaction_records/metrics/metric_{label_setting}_merged.json"
with open(merged_metric_save_path, 'w', encoding='utf-8') as f:
    json.dump(merged_metric, f, indent=4)