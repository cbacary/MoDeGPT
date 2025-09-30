import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("MoDeGPT")


def load_model(model_name: str, device: int = 0):
    """
    Load the original HuggingFace model and tokenizer,
    using float16 precision and specifying an explicit CUDA device.
    Do not use device_map='auto' to ensure stable loading.
    """
    try:
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

        model.eval()
        return model, tokenizer, model.config

    except Exception as e:
        logger.error(f"[Error] Failed to load model {model_name}: {e}")
        raise


def save_model(
    model: torch.nn.Module, tokenizer, save_dir: str, source_model_name: str
):
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


def reload_compressed_model(model_dir: str, device: int = 0):
    """
    Reload the compressed model and tokenizer,
    assuming that the model structure remains unchanged and only the parameters have been compressed.
    """
    try:
        logger.info(f"Reloading compressed model from: {model_dir}")
        tokenizer_source_path = os.path.join(model_dir, "tokenizer_source.txt")

        if not os.path.exists(tokenizer_source_path):
            raise FileNotFoundError(
                "Missing tokenizer_source.txt. Cannot reload tokenizer."
            )

        with open(tokenizer_source_path, "r") as f:
            tokenizer_source = f.read().strip()

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.to(f"cuda:{device}")
        logger.info(f"✔ Reloaded compressed model to cuda:{device} successfully.")

        model.eval()
        return model, tokenizer

    except Exception as e:
        logger.error(f"[Error] Failed to reload compressed model from {model_dir}: {e}")
        raise


def get_model_attrs(model):
    config = model.config
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

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        arch = "gpt"
    elif hasattr(model, "model") and hasattr(model.model, "decoder"):
        arch = "opt"
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        arch = "llama"
    else:
        raise RuntimeError("Unsupported model architecture")

    return n_layers, n_heads, d_model, head_dim, arch
