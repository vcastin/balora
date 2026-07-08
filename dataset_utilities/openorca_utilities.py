"""
Preprocessing utilities for OpenOrca dataset.
Dataset: https://huggingface.co/datasets/Open-Orca/OpenOrca
Format: system_prompt, question, response columns
"""

from typing import Dict, Any, List
from functools import partial
from dataset_utilities.common_utilities import preprocess_instruction_dataset


def format_openorca_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert OpenOrca example to chat messages format.

    OpenOrca dataset has:
    - id: unique identifier (not used in preprocessing)
    - system_prompt: system instructions (optional, can be empty)
    - question: user question/instruction
    - response: assistant response

    Args:
        example: Dict with 'system_prompt', 'question', 'response' keys

    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    system_prompt = example.get("system_prompt", "")
    question = example["question"]
    response = example["response"]

    messages = []

    # Add system message if present and not empty
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    # Add user question and assistant response
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": response})

    return messages


def preprocess_batch(batch: Dict[str, List[Any]], tokenizer, max_length: int) -> Dict[str, List]:
    """
    Preprocess OpenOrca dataset batch.

    Args:
        batch: Dictionary with 'system_prompt', 'question', 'response' lists
        tokenizer: HuggingFace tokenizer with chat template support
        max_length: Maximum sequence length

    Returns:
        Dictionary with input_ids, attention_mask, labels lists
    """
    return preprocess_instruction_dataset(
        batch=batch,
        tokenizer=tokenizer,
        max_length=max_length,
        format_messages_fn=format_openorca_messages
    )
