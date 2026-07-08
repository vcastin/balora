from typing import Dict, Any
from transformers import AutoTokenizer
# Import the common DataCollator to avoid duplication
from dataset_utilities.common_utilities import DataCollatorForCausalLM


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
    labels[:prompt_len] = [-100] * prompt_len

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def preprocess_batch(batch, tokenizer, max_length):
        # batch: dictionary with arrays (datasets map works with batched True)
        prompts = []
        completions = []
        for q, a in zip(batch["question"], batch.get("answer", [""] * len(batch["question"]))):
            p = PROMPT_TEMPLATE.format(question=q.strip())
            c = " " + a.strip()
            prompts.append(p)
            completions.append(c)

        tokenized = {"input_ids": [], "attention_mask": [], "labels": []}
        for p, c in zip(prompts, completions):
            t = tokenize_and_build_labels(tokenizer, p, c, max_length)
            tokenized["input_ids"].append(t["input_ids"])
            tokenized["attention_mask"].append(t["attention_mask"])
            tokenized["labels"].append(t["labels"])

        return tokenized
