"""
Preprocessing utilities for AI2 ARC dataset.
Dataset: https://huggingface.co/datasets/allenai/ai2_arc
Format: Multiple choice questions with id, question, choices, answerKey
Subsets: ARC-Easy, ARC-Challenge
"""

from typing import Dict, Any, List
from functools import partial
from dataset_utilities.common_utilities import preprocess_instruction_dataset


def format_ai2_arc_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert AI2 ARC example to chat messages format.

    Args:
        example: Dict with 'question', 'choices', 'answerKey' keys
                choices is a dict with 'text' (list of options) and 'label' (list of labels)

    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    question = example["question"]
    choices = example["choices"]
    answer_key = example["answerKey"]

    # Format the question with multiple choice options
    choice_texts = choices["text"]
    choice_labels = choices["label"]

    # Build the formatted question with choices
    formatted_choices = []
    for label, text in zip(choice_labels, choice_texts):
        formatted_choices.append(f"{label}. {text}")

    choices_str = "\n".join(formatted_choices)
    user_message = f"{question}\n\n{choices_str}"

    # Find the answer text corresponding to the answer key
    answer_idx = choice_labels.index(answer_key)
    answer_text = choice_texts[answer_idx]
    assistant_message = f"{answer_key}. {answer_text}"

    return [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]


def preprocess_batch(batch: Dict[str, List[Any]], tokenizer, max_length: int) -> Dict[str, List]:
    """
    Preprocess AI2 ARC dataset batch.

    Args:
        batch: Dictionary with 'question', 'choices', 'answerKey' lists
        tokenizer: HuggingFace tokenizer with chat template support
        max_length: Maximum sequence length

    Returns:
        Dictionary with input_ids, attention_mask, labels lists
    """
    return preprocess_instruction_dataset(
        batch=batch,
        tokenizer=tokenizer,
        max_length=max_length,
        format_messages_fn=format_ai2_arc_messages
    )
