"""
Preprocessing utilities for Alpaca dataset.
Dataset: https://huggingface.co/datasets/tatsu-lab/alpaca
Format: instruction, input, output columns
"""

from typing import Dict, Any, List
from functools import partial
from dataset_utilities.common_utilities import preprocess_instruction_dataset


def format_alpaca_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert Alpaca example to chat messages format.

    Args:
        example: Dict with 'instruction', 'input', 'output' keys

    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]

    # Combine instruction and input if input is not empty
    if input_text:
        user_message = f"{instruction}\n\n{input_text}"
    else:
        user_message = instruction

    return [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": output}
    ]


def preprocess_batch(batch: Dict[str, List[Any]], tokenizer, max_length: int) -> Dict[str, List]:
    """
    Preprocess Alpaca dataset batch.

    Args:
        batch: Dictionary with 'instruction', 'input', 'output' lists
        tokenizer: HuggingFace tokenizer with chat template support
        max_length: Maximum sequence length

    Returns:
        Dictionary with input_ids, attention_mask, labels lists
    """
    return preprocess_instruction_dataset(
        batch=batch,
        tokenizer=tokenizer,
        max_length=max_length,
        format_messages_fn=format_alpaca_messages
    )
