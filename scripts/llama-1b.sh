# Currently need to change sequence length to 131072 for this model 
#   (Token indices  sequence length is longer than the specified maximum sequence length for this model 
#   (250588 > 131072). Running this sequence through the model will result in indexing errors)
python run_modegpt.py \
  --model meta-llama/Llama-3.2-1B \
  --compression_ratio 0.8 \
  --calib_size 32 \
  --eval_size all \
  --output_dir ./compressed_output/llama3_2-1B_0.8 \
  --device 0
