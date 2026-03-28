TESTS_OUTPUT_DIR="$TMPDIR/compressed_output/Llama3-8B/"
MODEL_DIR="/blue/sgao1/cc22bc.fsu/prog/NySVD-MoE/compressed_output_backup/model/"

export HF_HUB_ENABLE_HF_TRANSFER=0
export TRITON_CACHE_DIR="/blue/sgao1/cc22bc.fsu/triton_cache/"
export TORCHINDUCTOR_CACHE_DIR="/blue/sgao1/cc22bc.fsu/inductor_cache/"

echo "Test 1"
python -m src.run_modegpt \
    --model "meta-llama/Meta-Llama-3-8B" \
    --device 0 \
    --compression_ratio 0.25 \
    --calib_size 128 \
    --calibs_batch_size 16 \
    --output_dir "$TESTS_OUTPUT_DIR" \
    --note "llama3-8b 128-calib" \
    --order "mlp,qk,vo" \
    --max_sparsity 0.8 \
    --ridge_vo 1e-4 \
    --ridge_qk 3e-4 \
    --sparsity_smoothing 0.071 \
    --dataset alpaca
