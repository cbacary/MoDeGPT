import argparse
import logging
import os

import torch

from calibration import load_calibs
from compress_cr import compress_qk
from compress_nystrom import compress_mlp
from compress_svd import compress_vo
from compression_utils import allocate_global_sparsity
from eval import compute_perplexity, load_calibration_texts, load_eval_texts
from model_utils import load_model, save_model

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
    parser.add_argument("--load_calibs_from", type=str, default="")
    parser.add_argument("--calibs_save_path", type=str, default="")

    args = parser.parse_args()

    device = args.device
    logger.info(f"Loading model: {args.model}")
    model, tokenizer, config = load_model(args.model, device=device)

    logger.info("Loading calibration and evaluation texts...")
    calib_texts = load_calibration_texts(args.calib_size, model, tokenizer)
    eval_texts = load_eval_texts(args.eval_size)

    logger.info("Evaluating original model (for baseline perplexity)...")
    baseline_ppl = compute_perplexity(model, tokenizer, eval_texts, device=device)
    logger.info(f"Original model perplexity on WikiText2: {baseline_ppl:.2f}")

    cov_mlp, cov_q, cov_k, cov_x, bi_scores = load_calibs(
        model,
        tokenizer,
        calib_texts,
        load_calibs_from=args.load_calibs_from,
        calibs_save_path=args.calibs_save_path,
    )

    layer_keep_ratios = allocate_global_sparsity(
        bi_scores, compression_ratio=args.compression_ratio
    )

    slice_dims = True
    ridge_lambda = 1e-2

    skip, local_path = args.skip, args.local_model_path
    if skip and local_path:
        logger.info(f"Loading local model path {local_path}")
        model, tokenizer, config = load_model(local_path)

    torch.cuda.empty_cache()
    model.cuda()

    logger.info("Beginning compression...")
    if "mlp" not in skip:
        compress_mlp(
            model=model,
            cov=cov_mlp,
            keep_ratios=layer_keep_ratios,
            ridge_lambda=1e-3,
            slice_dims=True,
        )

        save_model(
            model,
            tokenizer,
            save_dir=f"{args.output_dir}--mlp",
            source_model_name=args.model,
        )

    if "qk" not in skip:
        compress_qk(
            model=model,
            cov=(cov_q, cov_k),
            keep_ratios=layer_keep_ratios,
            ridge_lambda=ridge_lambda,
            slice_dims=True,
        )

        save_model(
            model,
            tokenizer,
            save_dir=f"{args.output_dir}--qk",
            source_model_name=args.model,
        )

    if "vo" not in skip:
        compress_vo(
            model=model,
            cov=cov_x,
            keep_ratios=layer_keep_ratios,
            ridge_lambda=ridge_lambda,
            slice_dims=True,
        )

        save_model(
            model,
            tokenizer,
            save_dir=f"{args.output_dir}--vo",
            source_model_name=args.model,
        )

    # patch_forward_OPT(
    #     model,
    # )
    model.cuda()
    compressed_ppl = compute_perplexity(model, tokenizer, eval_texts, device=device)
    logger.info(f"Compressed model perplexity on WikiText2: {compressed_ppl:.2f}")


main()
