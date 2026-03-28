import logging
import os

import torch

from src.model_utils import start_memory_usage_worker
from src.calibration import load_calibs

from src.compression.compress_mlp import compress_nystrom

from src.compression.compress_qk import compress_qk
from src.compression.compress_vo import compress_vo

from src.compression_utils import (
    allocate_global_sparsity,
)
from src.eval import (
    compute_perplexity,
)
from src.model_utils import (
    reload_compressed_model,
    save_compressed_model,
)

from src.adapters.model_adapter import ModelAdapter
from src.adapters.CompressionConfig import CompressionConfig

import optuna

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

import transformers.modeling_utils as tm
import torch.nn as nn

# 1. Target the correct function from your stack trace
_orig_load_param = tm._load_parameter_into_model


def _debug_load_param(*args, **kwargs):
    try:
        return _orig_load_param(*args, **kwargs)
    except Exception as e:  # Catch RuntimeError
        print("\n" + "!" * 60)
        print("🚨 CRASH DURING PARAMETER LOADING 🚨")

        # In this transformers function, the parameter name is usually the 2nd argument
        param_name = kwargs.get("param_name", kwargs.get("tensor_name", None))
        if not param_name and len(args) > 1:
            param_name = args[1]

        print(f"Failed on global parameter name: {param_name}")
        print(f"Original Error: {e}")
        print("!" * 60 + "\n")
        raise


# Apply the patch
tm._load_parameter_into_model = _debug_load_param


def main(trial: optuna.Trial = None, config: CompressionConfig | None = None):
    global decay_scores

    adapter = None
    start_memory_usage_worker()

    if not config:
        config = CompressionConfig.from_args()

    print(config.to_dict())

    model, tokenizer = reload_compressed_model(config.model)

    adapter = ModelAdapter.from_model(model=model, tokenizer=tokenizer)
    adapter.config = config

    baseline_ppl = compute_perplexity(
        model,
        tokenizer,
        dataset=adapter.config.dataset,
        adapter=adapter,
    )

    logger.info(f"Baseline ppl: {baseline_ppl}")
    adapter.metrics["baseline-ppl"] = baseline_ppl

    cov_mlp, cov_q, cov_k, cov_x, bi_scores = load_calibs(
        adapter=adapter,
        n_samples=config.calib_size,
        batch_size=config.calibs_batch_size,
        dataset=adapter.config.dataset,
    )

    layer_keep_ratios = allocate_global_sparsity(
        bi_scores,
        compression_ratio=config.compression_ratio,
        smoothing=adapter.config.sparsity_smoothing,
        max_sparsity=adapter.config.max_sparsity,
        adapter=adapter,
    )

    torch.cuda.empty_cache()

    order = config.order
    save_dir = os.path.join(config.output_dir, "model")

    if "mlp" in order:
        compress_nystrom(
            adapter=adapter,
            cov=cov_mlp,
            keep_ratios=layer_keep_ratios,
            target_layers=[i for i in range(adapter.n_layers)],
        )

    rotary_masks = None
    if "qk" in order:
        rotary_masks = compress_qk(
            adapter=adapter, cov=(cov_q, cov_k), keep_ratios=layer_keep_ratios
        )

    if "vo" in order:
        compress_vo(
            adapter=adapter,
            cov=cov_x,
            keep_ratios=layer_keep_ratios,
        )

    adapter.patch_config()

    save_compressed_model(
        adapter,
        rotary_masks=rotary_masks,
        save_dir=save_dir,
        source_model_name=adapter.config.model,
    )

    del model
    del tokenizer
    del cov_k
    del cov_q
    del cov_x
    del cov_mlp

    torch.cuda.empty_cache()
    import gc

    gc.collect()

    model, tokenizer = reload_compressed_model(save_dir)

    adapter.model = model
    adapter.tokenizer = tokenizer

    compressed_ppl = compute_perplexity(
        model,
        tokenizer,
        dataset=adapter.config.dataset,
        adapter=adapter,
    )
    adapter.metrics[f"ppl-{adapter.config.dataset}"] = compressed_ppl
    logger.info(f"Compressed (PPL): {compressed_ppl}")

    return compressed_ppl


main()
