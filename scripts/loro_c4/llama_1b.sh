# bash scripts/loro_c4/llama_1b.sh

cuda_idx=0,1,2,3
cuda_num=4

echo "Using multiple GPU"

rank=512
freq=200

CUDA_VISIBLE_DEVICES=$cuda_idx torchrun --standalone --nproc_per_node $cuda_num run_c4.py \
    --model_config configs/llama_1b.json \
    --dtype bfloat16 \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 100000 \
    --save_every 10000 \
    --eval_every 10000 \
    --lr 0.005 \
    --scheduler "cosine_restart" \
    --warmup_steps 10000 \
    --min_lr_ratio 0.1 \
    --cosine_restart_freq $freq \
    --lr_adjust_steps -10000 \
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
