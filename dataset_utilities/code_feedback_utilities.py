"""
Preprocessing utilities for CodeFeedback dataset.
Format: 'query' and 'answer' columns converted to chat messages format.
"""

from typing import Dict, Any, List
from functools import partial
from dataset_utilities.common_utilities import preprocess_instruction_dataset


def format_code_feedback_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert CodeFeedback example (query/answer) to chat messages format.

    Args:
        example: Dict containing 'query' and 'answer' keys.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """

    return [
        {"role": "user", "content": str(example.get("query", ""))},
        {"role": "assistant", "content": str(example.get("answer", ""))}
    ]


def preprocess_batch(batch: Dict[str, List[Any]], tokenizer, max_length: int) -> Dict[str, List]:
    """
    Preprocess CodeFeedback dataset batch.

    Args:
        batch: Dictionary with 'query' and 'answer' lists
        tokenizer: HuggingFace tokenizer with chat template support
        max_length: Maximum sequence length

    Returns:
        Dictionary with input_ids, attention_mask, labels lists
    """
    return preprocess_instruction_dataset(
        batch=batch,
        tokenizer=tokenizer,
        max_length=max_length,
        format_messages_fn=format_code_feedback_messages
    )

# """
# Preprocessing utilities for CodeFeedback dataset.
# Format: messages column with pre-formatted chat messages
# """

# from typing import Dict, Any, List
# from functools import partial
# from dataset_utilities.common_utilities import preprocess_instruction_dataset


# def format_code_feedback_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
#     """
#     Convert CodeFeedback example to chat messages format.

#     Args:
#         example: Dict with 'messages' key containing pre-formatted chat messages

#     Returns:
#         List of message dicts with 'role' and 'content' keys
#     """
#     # CodeFeedback already has messages in the correct format
#     return example["messages"]


# def preprocess_batch(batch: Dict[str, List[Any]], tokenizer, max_length: int) -> Dict[str, List]:
#     """
#     Preprocess CodeFeedback dataset batch.

#     Args:
#         batch: Dictionary with 'messages' list
#         tokenizer: HuggingFace tokenizer with chat template support
#         max_length: Maximum sequence length

#     Returns:
#         Dictionary with input_ids, attention_mask, labels lists
#     """
#     return preprocess_instruction_dataset(
#         batch=batch,
#         tokenizer=tokenizer,
#         max_length=max_length,
#         format_messages_fn=format_code_feedback_messages
#     )
