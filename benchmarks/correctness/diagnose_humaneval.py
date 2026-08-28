"""
Quick diagnostic: sends one HumanEval problem to the running API server
and prints the raw model output so we can see why pass@1 = 0.

Usage:
    python benchmarks/correctness/diagnose_humaneval.py
"""
import json
import requests

BASE_URL = "http://localhost:8000/v1/chat/completions"
MODEL    = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"

# HumanEval problem #0 (the simplest one)
HUMANEVAL_0_PROMPT = '''\
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''

HUMANEVAL_0_TEST = """\
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False
"""

def call_server(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "steps": 128,
    }
    resp = requests.post(BASE_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def try_execute(code: str, test: str, label: str) -> bool:
    full = code + "\n\n" + test + "\ncheck(has_close_elements)"
    try:
        exec(compile(full, "<string>", "exec"), {})
        print(f"  ✅ [{label}] PASSED the test case!")
        return True
    except Exception as e:
        print(f"  ❌ [{label}] FAILED: {e}")
        return False


def extract_code_from_markdown(text: str) -> str:
    """Extract code from markdown code blocks if present."""
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        return text[start:end].strip()
    return text


def main():
    print("=" * 60)
    print("  HumanEval Diagnostic — Testing server output format")
    print("=" * 60)
    print(f"\nSending HumanEval #0 to: {BASE_URL}\n")

    raw_output = call_server(HUMANEVAL_0_PROMPT)

    print("=" * 60)
    print("  RAW MODEL OUTPUT:")
    print("=" * 60)
    print(raw_output)
    print("=" * 60)

    # Attempt 1: Use raw output directly as function body (appended to prompt)
    full_raw = HUMANEVAL_0_PROMPT + raw_output
    print("\n--- Attempt 1: Raw output appended to prompt ---")
    try_execute(full_raw, HUMANEVAL_0_TEST, "raw")

    # Attempt 2: Extract code from markdown blocks
    extracted = extract_code_from_markdown(raw_output)
    if extracted != raw_output:
        print(f"\n--- Attempt 2: Extracted from markdown block ---")
        print(f"  Extracted:\n{extracted}")
        full_extracted = HUMANEVAL_0_PROMPT + extracted
        try_execute(full_extracted, HUMANEVAL_0_TEST, "extracted")

    # Attempt 3: Raw output as standalone (model may have re-defined the function)
    print(f"\n--- Attempt 3: Raw output as standalone function ---")
    try_execute(raw_output, HUMANEVAL_0_TEST, "standalone")

    print("\n" + "=" * 60)
    print("  CONCLUSION:")
    print("  If ALL 3 attempts fail, the model is not generating valid Python.")
    print("  If Attempt 2 or 3 passes, we need a post-processing step.")
    print("=" * 60)


if __name__ == "__main__":
    main()
