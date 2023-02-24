#!/bin/bash


echo "Start KoDiffCSE-SRoBERTa 0802 last mode"

python train.py \
    --model_name_or_path jhgan/ko-sroberta-multitask \
    --generator_name klue/roberta-small \
    --train_file /home/keonwoo/anaconda3/envs/KoDiffCSE/data/wiki_nli_sum.txt \
    --output_dir /home/keonwoo/anaconda3/envs/KoDiffCSE/last_training_0802 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --learning_rate 1e-6 \
    --max_seq_length 64 \
    --evaluation_strategy steps \
    --metric_for_best_model eval_spearman_cosine \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --logging_first_step \
    --logging_dir /home/keonwoo/anaconda3/envs/KoDiffCSE/log \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --batchnorm \
    --lambda_weight 0.005 \
    --fp16 --masking_ratio 0.30
