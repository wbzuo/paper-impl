#!/bin/bash

# 设置默认参数
SAVE_DIR="../out"
SAVE_WEIGHT="dpo"
EPOCHS=1
BATCH_SIZE=4
LEARNING_RATE=4e-8
DEVICE="cuda:0"
if ! nvidia-smi &> /dev/null; then
  DEVICE="cpu"
fi
DTYPE="bfloat16"
NUM_WORKERS=1
ACCUMULATION_STEPS=1
GRAD_CLIP=1.0
LOG_INTERVAL=100
SAVE_INTERVAL=100
HIDDEN_SIZE=512
NUM_HIDDEN_LAYERS=8
MAX_SEQ_LEN=1024
USE_MOE=1
DATA_PATH="../dataset/dpo.jsonl"
FROM_WEIGHT="full_sft"
FROM_RESUME=0
BETA=0.1
USE_WANDB=true
WANDB_PROJECT="minimind"

# 创建保存目录
mkdir -p $SAVE_DIR

# 运行训练脚本
torchrun --nproc_per_node 8 train_dpo.py \
  --save_dir $SAVE_DIR \
  --save_weight $SAVE_WEIGHT \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --device $DEVICE \
  --dtype $DTYPE \
  --num_workers $NUM_WORKERS \
  --accumulation_steps $ACCUMULATION_STEPS \
  --grad_clip $GRAD_CLIP \
  --log_interval $LOG_INTERVAL \
  --save_interval $SAVE_INTERVAL \
  --hidden_size $HIDDEN_SIZE \
  --num_hidden_layers $NUM_HIDDEN_LAYERS \
  --max_seq_len $MAX_SEQ_LEN \
  --use_moe $USE_MOE \
  --data_path $DATA_PATH \
  --from_weight $FROM_WEIGHT \
  --from_resume $FROM_RESUME \
  --beta $BETA \
  --use_wandb \
  --wandb_project $WANDB_PROJECT
