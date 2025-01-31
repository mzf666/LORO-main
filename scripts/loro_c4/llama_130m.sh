# bash scripts/loro_c4/llama_130m.sh

cuda_idx=0,3
cuda_num=2

echo "Using multiple GPU"

rank=256
freq=500

CUDA_VISIBLE_DEVICES=$cuda_idx torchrun --standalone --nproc_per_node $cuda_num run_c4.py \
    --model_config configs/llama_130m.json \
    --dtype bfloat16 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --save_every 2000 \
    --eval_every 2000 \
    --lr 0.01 \
    --scheduler "cosine_restart" \
    --warmup_steps 2000 \
    --min_lr_ratio 0.1 \
    --cosine_restart_freq $freq \
    --lr_adjust_steps -2000 \
    --weight_decay 0 \
    --optimizer loro_adamw \
    --loro_refresh all \
    --loro_refresh_freq $freq \
    --loro_scope all \
    --loro_init xavier \
    --loro_attn_rank $rank \
    --loro_mlp_rank $rank \
    --loro_type loro \
    --loro_freq $freq \
    --loro_lr_scaler -1
