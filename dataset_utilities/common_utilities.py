"""
Common preprocessing utilities for instruction-following datasets.
This module provides shared functionality for datasets that use chat templates.
"""

from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from transformers import AutoTokenizer
import torch


TRUNCATION = True


def preprocess_instruction_dataset(
    batch: Dict[str, List[Any]],
    tokenizer: AutoTokenizer,
    max_length: int,
    format_messages_fn: Callable[[Dict[str, Any]], List[Dict[str, str]]]
) -> Dict[str, List]:
    """
    Generic preprocessing function for instruction-following datasets.

    This function handles the common pattern of:
    1. Converting dataset format to chat messages
    2. Applying chat template
    3. Tokenizing
    4. Masking prompt tokens in labels

    Args:
        batch: Dictionary with lists of data (from datasets.map with batched=True)
        tokenizer: HuggingFace tokenizer with chat template support
        max_length: Maximum sequence length for truncation
        format_messages_fn: Function that takes a single example dict and returns
                          a list of chat messages with 'role' and 'content' keys

    Returns:
        Dictionary with input_ids, attention_mask, and labels lists
    """
    tokenized = {"input_ids": [], "attention_mask": [], "labels": []}

    # Get number of examples in batch
    num_examples = len(batch[list(batch.keys())[0]])

    # Process each example
    for i in range(num_examples):
        # Extract single example from batch
        example = {key: batch[key][i] for key in batch.keys()}

        # Convert to chat messages using provided function
        messages = format_messages_fn(example)

        # Apply chat template to format the full conversation
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Get user part (everything except last assistant message) for masking
        user_part = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        )

        full_tokens_test = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        
        if len(full_tokens_test) > max_length:
            continue

        tokenized_full = tokenizer(full_text, truncation=True, max_length=max_length)
        tokenized_user = tokenizer(user_part, truncation=True, max_length=max_length)

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        
        # Create labels - mask everything except assistant's response
        labels = list(input_ids)
        user_len = len(tokenized_user["input_ids"])

        # Mask the prompt (user + system tokens)
        mask_limit = min(user_len, len(input_ids))
        for j in range(mask_limit):
            labels[j] = -100

        tokenized["input_ids"].append(input_ids)
        tokenized["attention_mask"].append(attention_mask)
        tokenized["labels"].append(labels)

    return tokenized


@dataclass
class DataCollatorForCausalLM:
    """
    Data collator for causal language modeling.
    Pads sequences to the same length and converts to tensors.
    """
    tokenizer: AutoTokenizer
    padding: bool = True

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into a padded batch.

        Args:
            features: List of dicts with input_ids, attention_mask, labels

        Returns:
            Dictionary with padded tensors
        """
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded
        }
