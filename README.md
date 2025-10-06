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

uv venv --python 3.11
uv pip install -r requirements.txt
```

### 2. Run compression
OPT example usage skipping mlp and qk compression stages with 128 calibration samples and 4 eval samples. replace `--load_calibs_from` with `--calibs_save_path` to save calibration data instead of load:

```bash
python run_modegpt.py
  --model facebook/opt-1.3b   
  --compression_ratio 0.4   
  --calib_size 128   
  --eval_size 4   
  --output_dir ./compressed_output/opt-1.3b-0.3   
  --device 0  
  --skip mlp,qk
  --calibs_batch_size 4 
  --load_calibs_from ./calibs/calibs-sz128.pt
```


### 3. Additional Information

Currently only tested on OPT models. QK compression currently underperforms the paper performance, but all other stages approximate the papers' performance very well.

To implement for llama models you will have to modify the qk compression stage (`compress_cr.py`) to support the rotary positional embeddings (RoPE). Additionally, llama models use an additional gate matrix for the mlp compression (`compress_nystrom.py`) which will have to be added.

Additionally, you will have to create your own patched version of modeling_llama. See `patchers/OPTRebuild.py` and the `patch_config` function in `patchers/opt_patch.py` which will also be required to save and load the compressed model. This is a pretty easy process that just requires changing the dimensions of the linear layers when they are initialized to the compressed dimensions. 