import os
import json
import string
from typing import *
from IPython import embed


import torch
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm

from transformers.tokenization_utils import PreTrainedTokenizer
import copy
import random
    
IGNORE_INDEX=-100

def collator(tokenizer, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    attention_mask = torch.stack(attention_mask)
    # input_ids = torch.nn.utils.rnn.pad_sequence(
    #     input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    # )
    # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )


class PromptIterableDataset(IterableDataset):
    def __init__(self,
                 raw_dataset: Union[Dataset, List],
                 sep: List = ["EOS", "\n"],
                 tokenizer: PreTrainedTokenizer = None,
                 max_seq_length: Optional[int] = 512,
                 teacher_forcing: Optional[bool] = True,
                 truncate_method: Optional[str] = "tail",
                ):
        assert hasattr(raw_dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {raw_dataset}"
        assert hasattr(raw_dataset, "__len__"), f"The dataset must have __len__ method. dataset is {raw_dataset}"
        self.raw_dataset = raw_dataset
        self.sep = sep
        self._end_token = None
        self.start_token = self.sep[-1]
        self.teacher_forcing = teacher_forcing
        assert self.teacher_forcing, print("must use teacher forcing")

        self.tokenizer = tokenizer
        self.truncate_method = truncate_method
        self.max_seq_length = max_seq_length
        assert self.truncate_method == "tail", print("only tail truncate support")

    
    @property
    def end_token(self):
        if self._end_token is not None:
            return self._end_token
        end_token = self.sep[0]
        if end_token == "EOS":
            self._end_token = self.tokenizer.eos_token
        else:
            self._end_token = end_token
        return self._end_token

    def tokenize_example(self, example):
        end_token = self.end_token
        if len(example["data"]) % 2 != 0:
            example["data"] = example["data"][:-1]
        tags = [i for _ in range(len(example["data"])//2) for i in ["User", "Agent"]]
        # if example["id"].startswith("reasoning-"):
            # assert len(example["data"]) == 2, print(example)
            # tags = ["Question", "Answer"]
        labels = []
        tokenized_ids = []
        for i, c in enumerate(example["data"]):
            c_new = tags[i] + ": " + c + end_token
            if i % 2 == 1:
                # model
                c_input = self.start_token + tags[i] + ": "
                tokenized = self.tokenizer(c_input, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += [IGNORE_INDEX] * len(tokenized["input_ids"])

                c_generate = c + end_token
                tokenized = self.tokenizer(c_generate, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += tokenized["input_ids"]

            else:
                # user
                if i == 0:
                    # no start token
                    c_new = self.tokenizer.bos_token + tags[i] + ": " + c
                else:
                    c_new = self.start_token + tags[i] + ": " + c
                tokenized = self.tokenizer(c_new, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += [IGNORE_INDEX] * len(tokenized["input_ids"])

        assert len(tokenized_ids) == len(labels)

        return {"input_ids": torch.LongTensor(tokenized_ids), "labels": torch.LongTensor(labels)}

    def pad_truncate(self, tokenized_example):
        old_len = len(tokenized_example["input_ids"])
        tokenized_example["attention_mask"] = torch.LongTensor([1]*len(tokenized_example["input_ids"]))
        if old_len > self.max_seq_length:
            for k in tokenized_example:
                tokenized_example[k] = tokenized_example[k][(old_len - self.max_seq_length):]
        elif old_len < self.max_seq_length:
            tokenized_example["input_ids"] = torch.cat([torch.LongTensor([self.tokenizer.pad_token_id]*(self.max_seq_length - old_len)), tokenized_example["input_ids"]])
            tokenized_example["labels"] = torch.cat([torch.LongTensor([IGNORE_INDEX]*(self.max_seq_length - old_len)), tokenized_example["labels"]])
            tokenized_example["attention_mask"] = torch.LongTensor([0]*(self.max_seq_length - old_len) + [1]*old_len)
        assert len(tokenized_example["input_ids"]) == len(tokenized_example["labels"]) == len(tokenized_example["attention_mask"]) == self.max_seq_length
        return tokenized_example


    def __iter__(self):
        for example in self.raw_dataset:
            tokenized_example = self.tokenize_example(example)
            tokenized_example = self.pad_truncate(tokenized_example)
            yield tokenized_example

    def __len__(self):
        return len(self.raw_dataset)


if __name__ == "__main__":
    # from transformers import AutoTokenizer, LlamaTokenizer
    # print("here")
    # tokenizer = LlamaTokenizer.from_pretrained("/data/llama/llama-7b")
    # tokenizer.add_special_tokens({'pad_token': "<pad>"})
    # tokenizer.padding_side = "left"
    # text = "hi, this is a short poece of text."
    # tokenized = tokenizer(text, padding="max_length", max_length=20)
    # print(tokenized["input_ids"])
    # raw_dataset = load_raw_data("../data/processed/part2_1.json")
    # reasoning_data = load_reasoning_data("/data/dataset/reasoning")
    # print("loading...")
    # print(reasoning_data[0])
    # print(len(reasoning_data))

    # zh_data = load_zh_data("/data/dataset/ultra_zh/filtered_all.jsonl")
    # print("loading...")
    # print(list(random.sample(zh_data, 20)))
    # print(len(zh_data))

    data = load_mmlu_style_data("/mnt/data/user/tc_agi/user/chenyulin/dataset/mmlu_style_en/qa_pairs.jsonl")
    print(list(random.sample(data, 20)))
    print(len(data))

    # sharegpt_dataset = load_sharegpt_data("/data/dataset/sharegpt_data/ShareGPT_2023.05.08v0_Wasteland_Edition.json")
    # print(sharegpt_dataset[0])
    # print("done")
    # dataset = PromptIterableDataset(sharegpt_dataset, tokenizer=tokenizer, max_seq_length=2048, teacher_forcing=True)
    # for data in dataset:
    #     print(data)
    #     print(tokenizer.decode(data["input_ids"][:1000]))
        
    #     model_output = data["input_ids"][:1000][data["labels"][:1000]!=-100]
    #     print("##### model output")
    #     print(tokenizer.decode(model_output))
    #     break
