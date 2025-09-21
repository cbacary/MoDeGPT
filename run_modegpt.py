# -*- coding: utf-8 -*-
import argparse
import logging
import math
import os

import torch
from datasets import load_dataset

from calibration import calibrate_model
from compression_type1 import compress_mlp
from compression_type2 import compress_qk
from compression_type3 import compress_vo

# from compression_type3_cc import compress_vo
from evaluation import compute_perplexity
from model_utils import load_model, reload_compressed_model, save_model
from sparsity_alloc import allocate_global_sparsity

# ----------------------------
# Logger Setup
# ----------------------------
logger = logging.getLogger("MoDeGPT")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    os.makedirs("logs", exist_ok=True)
    file = logging.FileHandler("logs/run_modegpt.log")
    file.setFormatter(formatter)
    logger.addHandler(file)


# ----------------------------
# Load Calibration/Evaluation Data
# ----------------------------
def load_calibration_texts(calib_size):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 0]
    return texts if calib_size == "all" else texts[: int(calib_size)]


def load_eval_texts(eval_size):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    texts = [t for t in dataset["validation"]["text"] if len(t.strip()) > 0]
    return texts if eval_size == "all" else texts[: int(eval_size)]


# ----------------------------
# MoDeGPT Compression Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    parser.add_argument("--calib_size", type=str, default="32")
    parser.add_argument("--eval_size", type=str, default="32")
    parser.add_argument("--output_dir", type=str, default="compressed_output")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--skip", type=str, default="")
    parser.add_argument("--local_model_path", type=str, default="")

    args = parser.parse_args()

    # ----------------------------
    # Load Model and Tokenizer
    # ----------------------------
    device = args.device
    logger.info(f"Loading model: {args.model}")
    model, tokenizer, config = load_model(args.model, device=device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token for tokenizer.")

    # ----------------------------
    # Evaluate Original Model (Baseline PPL)
    # ----------------------------
    logger.info("Loading calibration and evaluation texts...")
    calib_texts = load_calibration_texts(args.calib_size)
    eval_texts = load_eval_texts(args.eval_size)

    logger.info("Evaluating original model (for baseline perplexity)...")
    baseline_ppl = compute_perplexity(model, tokenizer, eval_texts, device=device)
    logger.info(f"Original model perplexity on WikiText2: {baseline_ppl:.2f}")

    # ----------------------------
    # Step 1: Calibration
    # ----------------------------
    logger.info("Calibrating model...")
    cov_mlp, cov_q, cov_k, cov_x, bi_scores = calibrate_model(
        model, tokenizer, calib_texts, device=device, logger=logger
    )
    # ----------------------------
    # Step 2: Allocate Layer-wise Sparsity from BI Scores
    # ----------------------------
    logger.info("Allocating layer sparsity...")
    layer_keep_ratios = allocate_global_sparsity(
        bi_scores, config, target_keep_ratio=args.compression_ratio, temperature=1.0
    )

    # ----------------------------
    # Step 3: Free GPU Memory
    # ----------------------------
    logger.info("Freeing original model from GPU before compression...")
    model.cpu()
    torch.cuda.empty_cache()

    # Use original model object for in-place compression
    compressed_model = model

    # Extract config parameters
    n_layers = (
        getattr(config, "n_layer", None)
        or getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
    )
    n_heads = getattr(config, "n_head", None) or getattr(
        config, "num_attention_heads", None
    )
    d_model = getattr(config, "hidden_size", None) or getattr(config, "dim", None)
    head_dim = d_model // n_heads
    logger.info(
        f"n_layers={n_layers}, n_heads={n_heads}, d_model={d_model}, head_dim={head_dim}"
    )

    # ----------------------------
    # Step 4: Apply MoDeGPT Compression (Type I–III)
    # ----------------------------
    logger.info("Applying MoDeGPT compression methods...")

    ridge_lambda = 1e-2  # or expose as argparse argument

    skip, local_path = args.skip, args.local_model_path
    if skip and local_path:
        compressed_model = load_model(local_path, device=device)

    if "mlp" not in skip:
        compress_mlp(
            model=compressed_model,
            cov=cov_mlp,
            keep_ratios=layer_keep_ratios,
            rank=None,
            ridge_lambda=ridge_lambda,
            logger=logger,
        )

        save_model(
            compressed_model,
            tokenizer,
            save_dir=args.output_dir,
            source_model_name=f"{args.model}--mlp",
        )

    if "qk" not in skip:
        compress_qk(
            model=compressed_model,
            cov=(cov_q, cov_k),
            keep_ratios=layer_keep_ratios,
            rank=None,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            ridge_lambda=ridge_lambda,
            logger=logger,
        )

        save_model(
            compressed_model,
            tokenizer,
            save_dir=args.output_dir,
            source_model_name=f"{args.model}--qk",
        )

    compress_vo(
        model=compressed_model,
        cov=cov_x,
        keep_ratios=layer_keep_ratios,
        rank=None,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        ridge_lambda=ridge_lambda,
        min_rank=64,
        max_condition_number=1e3,
        logger=logger,
    )

    # ----------------------------
    # Save Final Compressed Model
    # ----------------------------
    logger.info("Saving compressed model...")
    save_model(model, tokenizer, args.output_dir, source_model_name=args.model)

    del compressed_model
    torch.cuda.empty_cache()

    # ----------------------------
    # Reload for Evaluation
    # ----------------------------
    logger.info("Reloading compressed model for evaluation...")
    compressed_model, tokenizer = reload_compressed_model(
        args.output_dir, device=device
    )

    compressed_ppl = compute_perplexity(
        compressed_model, tokenizer, eval_texts, device=device
    )
    if not math.isfinite(compressed_ppl):
        logger.warning(
            "⚠ Compressed model perplexity is NaN or Inf! Compression may have failed."
        )
    logger.info(f"Compressed model perplexity on WikiText2: {compressed_ppl:.2f}")

    # # ----------------------------
    # # Optional: Zero-Shot Example
    # # ----------------------------
    # results = evaluate_zero_shot(compressed_model, tokenizer, device=device)
    # logger.info(f"Zero-shot example:\n{results.get('completion_example', 'N/A')}")


if __name__ == "__main__":
    main()
