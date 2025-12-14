import logging
import math

import numpy as np
import torch
from datasets import load_dataset

logger = logging.getLogger("MoDeGPT")


def load_calibration_texts(calib_size, model, tokenizer, batch_size: int):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 0]

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


def load_eval_texts(eval_size, model, tokenizer, batch_size, stride: int):
    eval_size = int(eval_size)
    batch_size = int(batch_size)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    texts = [t for t in dataset["validation"]["text"] if len(t.strip()) > 0]

    joined_texts = "\n\n".join(texts)
    chunked = chunk_eval_texts(
        model=model, tokenizer=tokenizer, long_texts=joined_texts, min_threshold=2048, stride=stride
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


def chunk_text(model, tokenizer, long_texts: str, min_threshold, stride: int = 0):
    input_ids = tokenizer(long_texts, truncation=False, return_tensors=None)["input_ids"]

    max_length = min(2048, model.config.max_position_embeddings)

    text_chunks = []

    for i in range(0, len(input_ids), max_length - stride):
        chunk_ids = input_ids[i : i + max_length]

        if len(chunk_ids) < min_threshold:
            logger.info("Skipping text chunk")
            continue

        decoded_chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        text_chunks.append(decoded_chunk)

    return text_chunks


def chunk_eval_texts(model, tokenizer, long_texts: str, min_threshold, stride: int):
    input_ids = tokenizer(long_texts, truncation=False, return_tensors=None)["input_ids"]

    max_length = min(2048, model.config.max_position_embeddings)

    text_chunks = []

    for i in range(0, len(input_ids), stride):
        chunk_ids = input_ids[i : i + max_length]

        if len(chunk_ids) < min_threshold:
            logger.info("Skipping text chunk")
            continue

        decoded_chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        text_chunks.append(decoded_chunk)

    return text_chunks


@torch.no_grad()
def compute_perplexity(model, tokenizer, texts, stride: int, device="cuda"):
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

        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        seq_len = inputs["input_ids"].size(1)
        context_len = seq_len - stride

        if context_len > 0:
            labels[:, :context_len] = -100  # loss ignores

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        valid_tokens = labels != -100
        num_tokens = valid_tokens.sum().item()
        print(num_tokens)

        if not torch.isfinite(loss) or num_tokens == 0:
            print("not torch.isfinite(loss)")
            continue

        total_tokens += num_tokens
        total_loss += loss.item() * num_tokens

    if total_tokens == 0:
        raise ValueError("No valid tokens to compute perplexity!")

    avg_nll = total_loss / total_tokens
    return math.exp(avg_nll)
