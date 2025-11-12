import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


logger = logging.getLogger("MoDeGPT")

"""
precision to use for nearly all operations
 - qk compression uses less memory, will ignore this and use f64
 - vo may too
"""
dtype_p = torch.float64

"""
True if we will be using multiple gpu's to compress
 - Inference/calibration will not be performed in parallel (though could be done)
 - Weights will be compressed on the 2nd gpu, while the first GPU holds the entire model
"""
parallel = False
d1 = "cuda:0"
d2 = "cuda:1" if parallel else "cuda:0"
calib_device = "cuda:1" if parallel else "cuda:0"

def load_model(model_name: str, patched_models_dir = "./patched_models/", device: int = 0):
    """
    Loads the patched version of the model. Loading of the original model should only have to 
    be done once
    """
    from patchers.patch import patch_config

    model_dir = os.path.join(patched_models_dir, model_name)
    if os.path.exists(model_dir):
        import shutil

        rebuild_path = "./patchers/LlamaRebuild.py"
        shutil.copy(rebuild_path, model_dir)
        model, tokenizer = reload_compressed_model(model_dir)

        return model, tokenizer, model.config

    os.makedirs(model_dir, exist_ok=True)

    logger.info(f"Loading model from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to(f"cuda:{device}")
    logger.info(f"✔ Loaded model on cuda:{device} with float16.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("No pad_token found. Set pad_token = eos_token.")

    patch_config(model)
    arch = model.config.model_type
    if arch == "opt":
        rebuild_path = "./patchers/OPTRebuild.py"
    elif arch == "llama":
        rebuild_path = "./patchers/LlamaRebuild.py"
    else:
        raise AttributeError("Must provide patched modeling script")

    save_compressed_model(model, tokenizer, None, rebuild_path, model_dir, model_name)
    del model
    del tokenizer
    torch.cuda.empty_cache()
    model, tokenizer = reload_compressed_model(model_dir)

    model.eval()
    return model, tokenizer, model.config

def save_model(model: torch.nn.Module, tokenizer, save_dir: str, source_model_name: str):
    """
    Save the compressed model and tokenizer without changing the model structure,
    only saving the weights.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)

        # Save model weights + config
        model.save_pretrained(save_dir, torch_dtype=torch.float16)
        tokenizer.save_pretrained(save_dir)

        # Save tokenizer source for reload
        with open(os.path.join(save_dir, "tokenizer_source.txt"), "w") as f:
            f.write(source_model_name.strip())

        logger.info(f"✔ Model, tokenizer, and tokenizer_source.txt saved to {save_dir}")

    except Exception as e:
        logger.error(f"[Error] Failed to save model to {save_dir}: {e}")
        raise


def save_compressed_model(
    model: torch.nn.Module,
    tokenizer,
    rotary_masks: torch.Tensor | None,
    rebuild_path: str,
    save_dir: str,
    source_model_name: str,
):
    import shutil

    os.makedirs(save_dir, exist_ok=True)

    if rotary_masks is not None:
        mask_path = os.path.abspath(os.path.join(save_dir, "rotary_masks.pt"))
        model.config.mask_path = mask_path
    else:
        model.config.mask_path = None

    model.save_pretrained(save_dir, torch_dtype=torch.float16)
    tokenizer.save_pretrained(save_dir)
    if rotary_masks is not None:
        torch.save(rotary_masks, mask_path)
    shutil.copy(rebuild_path, save_dir)
    with open(os.path.join(save_dir, "tokenizer_source.txt"), "w") as f:
        f.write(source_model_name.strip())

    logger.info(f"✔ Model, tokenizer, and tokenizer_source.txt saved to {save_dir}")


def reload_compressed_model(model_dir: str, device: int = 0):
    logger.info(f"Reloading compressed model from: {model_dir}")
    tokenizer_source_path = os.path.join(model_dir, "tokenizer_source.txt")

    if not os.path.exists(tokenizer_source_path):
        raise FileNotFoundError("Missing tokenizer_source.txt. Cannot reload tokenizer.")

    with open(tokenizer_source_path, "r") as f:
        tokenizer_source = f.read().strip()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    model.to(f"cuda:{device}")
    logger.info(f"✔ Reloaded compressed model to cuda:{device} successfully.")

    model.eval()
    return model, tokenizer


def get_model_attrs(model):
    """
    returns:
    `n_layers, n_heads, d_model, head_dim, arch`
    """
    config = model.config
    n_layers = (
        getattr(config, "n_layer", None)
        or getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
    )
    n_heads = getattr(config, "n_head", None) or getattr(config, "num_attention_heads", None)
    d_model = getattr(config, "hidden_size", None) or getattr(config, "dim", None)
    head_dim = d_model // n_heads
    logger.info(f"n_layers={n_layers}, n_heads={n_heads}, d_model={d_model}, head_dim={head_dim}")

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        arch = "gpt"
    elif hasattr(model, "model") and hasattr(model.model, "decoder"):
        arch = "opt"
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        arch = "llama"
    else:
        raise RuntimeError("Unsupported model architecture")

    return n_layers, n_heads, d_model, head_dim, arch
