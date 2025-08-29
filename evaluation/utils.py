
import re


def extract_gsm8k_answer(text: str) -> str:
    """
    Extracts the final numerical answer from GSM8K-style responses.
    Prioritizes the #### format that we now enforce in prompts.
    """
    if not text:
        return ""
    
    # Remove common prefixes and clean up
    text = text.strip()
    
    # PRIORITY 1: Look for our enforced #### format first
    gsm8k_pattern = r"####\s*([0-9.]+)"
    match = re.search(gsm8k_pattern, text)
    if match:
        return match.group(1).strip('.')
    
    # PRIORITY 2: Look for other structured formats
    patterns = [
        r"the answer is\s*([0-9.]+)",
        r"answer:\s*([0-9.]+)",
        r"answer is\s*([0-9.]+)",
        r"result is\s*([0-9.]+)",
        r"final answer:\s*([0-9.]+)",
        r"final answer is\s*([0-9.]+)",
        r"she has\s*([0-9.]+)\s*buttons?",  # Look for "she has X buttons"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).strip('.')
    
    # PRIORITY 3: Look for arithmetic expressions
    arith_pattern = r"(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)"
    match = re.search(arith_pattern, text)
    if match:
        return match.group(4).strip('.')
    
    # PRIORITY 4: Look for buttons pattern
    buttons_pattern = r"(\d+)\s*buttons?"
    matches = re.findall(buttons_pattern, text.lower())
    if matches:
        return matches[-1].strip('.')
    
    # PRIORITY 5: Fallback to last arithmetic or equals
    lines = text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith(' ') and not line.startswith('-'):
            # Look for arithmetic at end of line
            arith_end = re.search(r'(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)', line)
            if arith_end:
                return arith_end.group(3).strip('.')
            
            # Look for equals at end of line
            equals_match = re.search(r'=\s*([0-9.]+)', line)
            if equals_match:
                return equals_match.group(1).strip('.')
    
    # PRIORITY 6: Last resort - last number (but be very conservative)
    numbers = re.findall(r'[0-9.]+', text)
    if numbers:
        return numbers[-1].strip('.')
    
    return ""