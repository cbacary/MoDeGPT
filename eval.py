import logging
import math

import torch
from datasets import load_dataset

logger = logging.getLogger("MoDeGPT")


def load_calibration_texts(calib_size):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 0]
    return texts if calib_size == "all" else texts[: int(calib_size)]


def load_eval_texts(eval_size):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    texts = [t for t in dataset["validation"]["text"] if len(t.strip()) > 0]
    return texts if eval_size == "all" else texts[: int(eval_size)]


@torch.no_grad()
def compute_perplexity(model, tokenizer, texts, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    max_length = getattr(tokenizer, "model_max_length", None)
    if max_length is None or max_length > 4096:
        max_length = 2048

    full_text = "\n".join([t.strip() for t in texts if len(t.strip()) > 0])

    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=False,
        padding=False,
    ).to(device)

    input_ids = inputs["input_ids"][0]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for start_idx in range(0, len(input_ids), max_length):
        end_idx = min(start_idx + max_length, len(input_ids))
        input_chunk = input_ids[start_idx:end_idx].unsqueeze(0)

        if input_chunk.size(1) < 2:
            continue

        labels = input_chunk.clone()
        attention_mask = (input_chunk != tokenizer.pad_token_id).long()

        outputs = model(
            input_ids=input_chunk, labels=labels, attention_mask=attention_mask
        )
        loss = outputs.loss

        if not torch.isfinite(loss):
            continue

        total_loss += loss.item() * (input_chunk.size(1) - 1)
        total_tokens += input_chunk.size(1) - 1

    if total_tokens == 0:
        raise ValueError("No valid tokens to compute perplexity!")

    avg_nll = total_loss / total_tokens
    return math.exp(avg_nll)


@torch.no_grad()
def generate_text(
    model, tokenizer, prompt: str, max_length: int = 50, device: str = "cpu"
):
    """
    Generate greedy decoded text from prompt.
    """
    model.eval()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(device)

    input_ids = inputs["input_ids"]

    gen_ids = model.generate(
        input_ids,
        max_length=input_ids.size(1) + max_length,
        do_sample=False,  # Greedy
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def evaluate_zero_shot(model, tokenizer, tasks_data=None, device: str = "cpu"):
    """
    Perform a simple zero-shot evaluation on prompt completion (MoDeGPT-style example).
    """
    results = {}

    # Example task: text continuation (greedy)
    prompt = "The study concludes that"
    continuation = generate_text(model, tokenizer, prompt, max_length=30, device=device)
    results["completion_example"] = continuation

    # (Optional) More prompts from tasks_data can be added here for full evaluation
    return results
