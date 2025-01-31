# bash scripts/run_glue.sh

cuda_idx=0

CUDA_VISIBLE_DEVICES=$cuda_idx python run_glue.py \
    --model_name_or_path roberta-base \
    --task_name mrpc \
    --max_length 512 \
    --seed 0 \
    --optimizer "loro_adamw" \
    --per_device_train_batch_size 16 \
    --num_train_epochs 20 \
    --learning_rate 0.0002 \
    --lr_scheduler_type linear \
    --weight_decay 0 \
    --loro_type "loro" \
    --loro_rank 8 \
    --loro_alpha 8 \
    --loro_freq 100 \
    --loro_init "xavier" \
    --loro_scope "qv" \
    --loro_lr_scaler 1
