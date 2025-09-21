#$ -N modegpt_llama2_7b_run
#$ -V
#$ -cwd
#$ -o logs/llama2-7b_fin/llama2-7b.o$JOB_ID
#$ -e logs/llama2-7b_fin/llama-7b.e$JOB_ID
#$ -l h_rt=12:00:00
#$ -l h_vmem=64G
#$ -pe sharedmem 4
#$ -l A100,gpu,gpu_mem=80G,cuda=1


  python run_modegpt.py \
  --model meta-llama/Llama-2-7b-hf \
  --compression_ratio 0.7 \
  --calib_size 32 \
  --eval_size all \
  --output_dir /u/scratch/x/xxiong/compressed_output/llama2-7b_0.7 \
  --device 2
