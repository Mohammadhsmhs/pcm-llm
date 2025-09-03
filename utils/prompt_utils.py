"""
Prompt utility functions for structured output formatting.
"""

def add_structured_instructions(prompt: str, task_type: str) -> str:
    """
    Add task-specific structured output instructions to the prompt.

    Args:
        prompt: The original prompt
        task_type: The type of task (reasoning, summarization, classification)

    Returns:
        The prompt with structured instructions added
    """
    if task_type == "reasoning":
        return prompt + "\n\nPlease provide your final answer in this exact format: #### [final_answer]\nWhere [final_answer] is the final answer to the problem with no extra characthers."
    elif task_type == "classification":
        return prompt + "\n\nPlease provide your final answer in this exact format: #### [sentiment_label]\nWhere [sentiment_label] is either 'positive' or 'negative'."
    elif task_type == "summarization":
        return prompt + "\n\nPlease provide your summary in this exact format: #### [summary_text]\nWhere [summary_text] is no more than two or three sentence summary of the main points discussed."
    else:
        return prompt + "\n\nPlease provide your final answer in this exact format: #### [answer]"
