from typing import Dict, Any
from dataclasses import dataclass
from transformers import AutoTokenizer
import torch


SYSTEM_PROMPT = ""
PROMPT_TEMPLATE = "Question: {question}\nAnswer:"
TRUNCATION = True

def make_prompt(example: Dict[str, Any]) -> Dict[str, str]:
    question = example["question"].strip()
    answer = example.get("answer", "").strip()
    prompt = PROMPT_TEMPLATE.format(question=question)
    completion = " " + answer
    return {"prompt": prompt, "completion": completion}


def tokenize_and_build_labels(tokenizer, prompt, completion, max_length):
    input_text = prompt + completion
    tokenized_full = tokenizer(
        input_text,
        truncation=TRUNCATION,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    tokenized_prompt = tokenizer(
        prompt,
        truncation=TRUNCATION,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]
    prompt_len = len(tokenized_prompt["input_ids"])
    labels = input_ids.copy()
    actual_prompt_limit = min(prompt_len, len(input_ids))
    labels[:actual_prompt_limit] = [-100] * actual_prompt_limit

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def preprocess_batch(batch, tokenizer, max_length):
    queries = batch["query"]
    responses = batch.get("response", [""] * len(queries))
    
    tokenized = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for q, r in zip(queries, responses):
        p = PROMPT_TEMPLATE.format(question=q.strip())
        c = " " + r.strip() + tokenizer.eos_token
        
        full_tokens = tokenizer(p + c, add_special_tokens=False)["input_ids"]
        
        if len(full_tokens) <= max_length:
            t = tokenize_and_build_labels(tokenizer, p, c, max_length)
            tokenized["input_ids"].append(t["input_ids"])
            tokenized["attention_mask"].append(t["attention_mask"])
            tokenized["labels"].append(t["labels"])
        else:
            continue 

    return tokenized


@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer
    padding: bool = True

    def __call__(self, features):
        # features is a list of dicts with input_ids, attention_mask, labels (lists)
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {"input_ids": input_ids_padded, "attention_mask": attention_mask_padded, "labels": labels_padded}