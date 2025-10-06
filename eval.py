import logging
import math

import numpy as np
import torch
from datasets import load_dataset

logger = logging.getLogger("MoDeGPT")


def load_calibration_texts(calib_size, model, tokenizer, batch_size: int):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 0]

    # What i thought was possibly an issue:
    # joining different portions of the calibration text into one,
    # mangling order and thus affecting calibration
    # apparently not the issue. worse perplexiy with the following
    # s = np.random.choice(texts, int(calib_size), replace=False)
    # o = []
    # for i in range(0, len(s), batch_size):
    #     o.append(s[i : i + batch_size].tolist())

    # return o

    joined_texts = "\n".join(texts)
    chunked = chunk_text(
        model=model, tokenizer=tokenizer, long_texts=joined_texts, min_threshold=2048
    )

    np.random.seed(1234)
    batches = []
    for i in range(0, int(calib_size), batch_size):
        batches.append(
            np.random.choice(
                chunked,
                size=int(batch_size),
                replace=False,
            ).tolist()
        )

    return batches


def load_eval_texts(eval_size, model, tokenizer, batch_size):
    eval_size = int(eval_size)
    batch_size = int(batch_size)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    texts = [t for t in dataset["validation"]["text"] if len(t.strip()) > 0]

    joined_texts = "\n".join(texts)
    chunked = chunk_text(
        model=model, tokenizer=tokenizer, long_texts=joined_texts, min_threshold=2048
    )

    np.random.seed(1234)
    batches = []
    for i in range(0, eval_size, batch_size):
        batches.append(
            np.random.choice(
                chunked,
                size=int(batch_size),
                replace=False,
            ).tolist()
        )

    return batches

    # return texts if eval_size == "all" else texts[: int(eval_size)]


def chunk_text(model, tokenizer, long_texts: str, stride: int = 0, min_threshold: int = 10):
    input_ids = tokenizer(long_texts, truncation=False, return_tensors=None)["input_ids"]

    max_length = model.config.max_position_embeddings

    text_chunks = []

    for i in range(0, len(input_ids), max_length - stride):
        chunk_ids = input_ids[i : i + max_length]

        if len(chunk_ids) < min_threshold:
            logger.info("Skipping text chunk")
            continue

        decoded_chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        text_chunks.append(decoded_chunk)

    return text_chunks


@torch.no_grad()
def compute_perplexity(model, tokenizer, texts, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    max_length = getattr(tokenizer, "model_max_length", None)
    if max_length is None or max_length > 4096:
        max_length = 2048

    for count, batch in enumerate(texts):
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(device="cuda")
        print(f"inputs['input_ids'].shape = {inputs['input_ids'].shape}")
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        if not torch.isfinite(loss):
            print("not torch.isfinite(loss)")
            continue

        num_tokens = inputs["attention_mask"].sum().item()
        print(num_tokens)
        total_tokens += num_tokens
        total_loss += loss.item() * num_tokens

    if total_tokens == 0:
        raise ValueError("No valid tokens to compute perplexity!")

    avg_nll = total_loss / total_tokens
    return math.exp(avg_nll)


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_length: int = 50, device: str = "cpu"):
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
