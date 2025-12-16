# A code implemention of MoDeGPT

This repository is a code implemention of [MoDeGPT](https://arxiv.org/abs/2408.09632) in ICLR 2025

### 1. Set-up

1. Clone this repository

```bash
git clone https://github.com/cbacary/MoDeGPT.git
cd MoDeGPT
```

2. Install Package

using `uv`:

```Shell

uv venv --python 3.12
uv pip install -r requirements.txt
```

### 2. Run compression

Llama2-7b example usage skipping mlp compression stage with 128 calibration samples.

```bash
python run_modegpt.py
  --model meta-llama/Llama-2-7b-hf
  --compression_ratio 0.25
  --calib_size 128
  --eval_size 128
  --calibs_batch_size 16
  --output_dir ./compressed_output/llama2-7b
  --device 0
  --skip mlp
```

### 3. Additional Information

Currently tested against OPT, Llama2-7b, and llama3-8b models. Llama3-8b models provided better calibration and eval against the Aplaca dataset. For llama3-8b, you can set `ALPACA=True` at the top of `run_modegpt.py`

To implement for additional model architectures you will have to modify / create your own patched version of modeling_llama (different for each architecture). See `patchers/OPTRebuild.py`, `patchers/LlamaRebuild.py` and the `patch_config` function in `patchers/patch.py`. This process mostly involves changing the dimensions of the linear layers when they are initialized to the compressed dimensions, but other steps may be required depending on the architecture you are working with.
