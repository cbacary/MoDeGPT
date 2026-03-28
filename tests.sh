MODEL_NAME="Qwen/Qwen3-8B"
TESTS_OUTPUT_DIR="$TMPDIR/compressed_output/$MODEL_NAME/"

export HF_HUB_ENABLE_HF_TRANSFER=0

echo "Test 1"
python -m src.run_modegpt \
    --model "$MODEL_NAME" \
    --device 0 \
    --compression_ratio 0.30 \
    --calib_size 128 \
    --calibs_batch_size 16 \
    --output_dir "$TESTS_OUTPUT_DIR" \
    --note "llama3-8b 128-calib" \
    --order "mlp,qk,vo" \
    --max_sparsity 0.8 \
    --ridge_vo 1e-4 \
    --ridge_qk 1e-1 \
    --sparsity_smoothing 0.0725 \
    --dataset alpaca