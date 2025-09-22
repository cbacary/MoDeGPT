## Save calibs
python run_modegpt.py \
  --model facebook/opt-1.3b \
  --compression_ratio 0.8 \
  --calib_size 16 \
  --eval_size 16 \
  --output_dir ./compressed_output/opt-1.3b-custom_vo_test \
  --skip mlp,qk, \
  --local_model_path ./compressed_output/opt-1.3b--qk/ \
  --calibs_save_path ./calibs.pt \
  --device 0

## load from calibs
python run_modegpt.py \
  --model facebook/opt-1.3b \
  --compression_ratio 0.8 \
  --calib_size 16 \
  --eval_size 16 \
  --output_dir ./compressed_output/opt-1.3b-custom_vo_test \
  --skip mlp,qk, \
  --local_model_path ./compressed_output/opt-1.3b-full--qk/ \
  --load_calibs_from ./calibs.pt \
  --device 0
  # --skip mlp,\
  # --local_model_path ./compressed_output/llama3_2-1B_0.8--mlp/ \

python run_modegpt.py \
  --model facebook/opt-1.3b \
  --compression_ratio 0.8 \
  --calib_size 16 \
  --eval_size 16 \
  --output_dir ./compressed_output/opt-1.3b-full \
  --device 0