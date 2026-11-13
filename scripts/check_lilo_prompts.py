"""Check full Ultra-Long instructions against the unchanged pi05 tokenizer limit."""

import json

from scripts.lilo_protocol import task_definitions


def validate_prompts(tokenizer, maximum):
    rows = []
    for task in task_definitions():
        text = task["prompt"].strip().replace("_", " ").replace("\n", " ")
        count = len(tokenizer.encode(text, add_bos=True)) + len(tokenizer.encode("\n"))
        rows.append({"task_id": task["task_id"], "token_count": count, "limit": maximum})
    too_long = [r for r in rows if r["token_count"] > maximum]
    if too_long:
        raise ValueError("Full Ultra-Long instruction would be truncated; do not evaluate: " + json.dumps(too_long))
    return rows


def check_prompts():
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.training.config import get_config
    config = get_config("pi05_libero").model
    if config.discrete_state_input:
        raise ValueError("This prompt check expects pi05_libero's continuous state input")
    tokenizer = PaligemmaTokenizer(max_len=config.max_token_len)._tokenizer
    return validate_prompts(tokenizer, config.max_token_len)


if __name__ == "__main__":
    print(json.dumps({"status": "PASS", "prompts": check_prompts()}, indent=2))
