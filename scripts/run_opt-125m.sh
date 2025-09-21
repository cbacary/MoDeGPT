#!/bin/bash
#$ -V
#$ -N modegpt_opt_125m_run           
#$ -cwd                     
#$ -o logs/opt-125m.o$JOB_ID
#$ -e logs/opt-125m.e$JOB_ID
#$ -l h_rt=12:00:00         
#$ -l h_vmem=16G            
#$ -pe sharedmem 4          
#$ -l gpu=1                 


# run MoDeGPT
python run_modegpt.py \
  --model facebook/opt-125m \
  --compression_ratio 0.9 \
  --calib_size 8 \
  --eval_size all \
  --output_dir compressed_output