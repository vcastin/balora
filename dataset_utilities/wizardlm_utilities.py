"""
Preprocessing utilities for WizardLM dataset.
Dataset: https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_70k
Format: instruction, output columns
"""

from typing import Dict, Any, List
from functools import partial
from dataset_utilities.common_utilities import preprocess_instruction_dataset


def format_wizardlm_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert WizardLM example to chat messages format.

    Args:
        example: Dict with 'instruction', 'output' keys

    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    instruction = example["instruction"]
    output = example["output"]

    return [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output}
    ]


def preprocess_batch(batch: Dict[str, List[Any]], tokenizer, max_length: int) -> Dict[str, List]:
    """
    Preprocess WizardLM dataset batch.

    Args:
        batch: Dictionary with 'instruction', 'output' lists
        tokenizer: HuggingFace tokenizer with chat template support
        max_length: Maximum sequence length

    Returns:
        Dictionary with input_ids, attention_mask, labels lists
    """
    return preprocess_instruction_dataset(
        batch=batch,
        tokenizer=tokenizer,
        max_length=max_length,
        format_messages_fn=format_wizardlm_messages
    )
