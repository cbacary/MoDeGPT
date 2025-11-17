import argparse
import logging
import os

import torch

from model_utils import get_model_attrs
from calibration import load_calibs
from compress_qk import compress_qk, compress_qk_svd
from compress_mlp import compress_mlp
from compress_vo import compress_vo
from compression_utils import allocate_global_sparsity
from eval import compute_perplexity, load_calibration_texts, load_eval_texts
from model_utils import load_model, reload_compressed_model, save_compressed_model, save_model
from patchers.patch import patch_config



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
    parser.add_argument("--calibs_batch_size", type=int, default=4)

    args = parser.parse_args()

    device = args.device
    logger.info(f"Loading model: {args.model}")
    model, tokenizer, config = load_model(args.model, device=device)

    logger.info("Loading calibration and evaluation texts...")
    calib_texts = load_calibration_texts(
        args.calib_size, model, tokenizer, batch_size=int(args.calibs_batch_size)
    )
    eval_texts = load_eval_texts(
        args.eval_size, model, tokenizer, batch_size=args.calibs_batch_size
    )

    logger.info("Evaluating original model (for baseline perplexity)...")
    baseline_ppl = compute_perplexity(model, tokenizer, eval_texts, device=device)
    logger.info(f"Original model perplexity on WikiText2: {baseline_ppl:.2f}")

    cov_mlp, cov_q, cov_k, cov_x, bi_scores = load_calibs(
        model,
        tokenizer,
        calib_texts,
        int(args.calibs_batch_size),
        load_calibs_from=args.load_calibs_from,
        calibs_save_path=args.calibs_save_path,
    )

    layer_keep_ratios = allocate_global_sparsity(
        bi_scores, compression_ratio=args.compression_ratio
    )

    # model, tokenizer, config = load_model(args.model, device=device)

    slice_dims = True
    ridge_lambda = 1e-2

    skip, local_path = args.skip, args.local_model_path
    if skip and local_path:
        logger.info(f"Loading local model path {local_path}")
        model, tokenizer, config = load_model(local_path)

    n_layers, n_heads, d_model, head_dim, arch = get_model_attrs(model)
    logger.info(f"arch == {arch}")

    torch.cuda.empty_cache()
    model.cuda()
    model.eval()
    logger.info("Beginning compression...")
    if "mlp" not in skip:
        compress_mlp(
            model=model,
            cov=cov_mlp,
            keep_ratios=layer_keep_ratios,
            ridge_lambda=1e-3,
            slice_dims=True,
        )

    slice_vo_qk = True
    rotary_masks = None
    if "qk" not in skip:
        if arch == "opt":
            compress_qk_svd(
                model=model,
                cov_x=cov_x,
                keep_ratios=layer_keep_ratios,
                ridge_lambda=ridge_lambda,
                slice_dims=slice_vo_qk,
            )
        else:
            rotary_masks = compress_qk(
                model=model,
                cov=(cov_q, cov_k),
                keep_ratios=layer_keep_ratios,
                ridge_lambda=ridge_lambda,
                slice_dims=slice_vo_qk,
            )

    if "vo" not in skip:
        compress_vo(
            model=model,
            cov=cov_x,
            keep_ratios=layer_keep_ratios,
            ridge_lambda=ridge_lambda,
            slice_dims=slice_vo_qk,
        )

    patch_config(model)

    if arch == "opt":
        rebuild_path = "./patchers/OPTRebuild.py"
    elif arch == "llama":
        rebuild_path = "./patchers/LlamaRebuild.py"
    else:
        raise Exception("Cannot save compressed model ... no compressed model definition")

    save_compressed_model(
        model,
        tokenizer,
        rotary_masks,
        rebuild_path=rebuild_path,
        save_dir=args.output_dir,
        source_model_name=args.model,
    )
    

    del model
    del tokenizer
    del cov_k
    del cov_q
    del cov_x
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    model = "i hope this deletes it really"
    gc.collect()
    model, tokenizer = reload_compressed_model(args.output_dir, device="cuda")

    eval_texts = load_eval_texts(
        args.eval_size, model, tokenizer, batch_size=args.calibs_batch_size
    )
    # if slice_vo_qk:
    #     from compression_utils import patch_OPT

    #     patch_OPT(
    #         model,
    #     )
    # eval_texts = load_eval_texts(
    #     args.eval_size, model, tokenizer, batch_size=args.calibs_batch_size
    # )
    compressed_ppl = compute_perplexity(model, tokenizer, eval_texts)
    logger.info(f"Compressed model perplexity on WikiText2: {compressed_ppl:.2f}")


main()
