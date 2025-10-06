# A code implemention of MoDeGPT
This repository is a code implemention of [MoDeGPT](https://arxiv.org/abs/2408.09632) in ICLR 2025

### 1. Set-up
1. Clone this repository
```bash
git clone https://github.com/cbacary/MoDeGPT.git
cd MoDeGPT
```
2. Install Package 
**uv** or whatever env manager you use
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