#!/bin/bash

# Set default values for the arguments
SAVE_DIR="../out"
SAVE_WEIGHT="full_sft"
EPOCHS=2
BATCH_SIZE=64
LEARNING_RATE=5e-7
DEVICE=$(if [ $(nvidia-smi -L | wc -l) -gt 0 ]; then echo "cuda:0"; else echo "cpu"; fi)
DTYPE="bfloat16"
NUM_WORKERS=1
ACCUMULATION_STEPS=1
GRAD_CLIP=1.0
LOG_INTERVAL=100
SAVE_INTERVAL=100
HIDDEN_SIZE=512
NUM_HIDDEN_LAYERS=8
MAX_SEQ_LEN=512
USE_MOE=1
DATA_PATH="../dataset/sft_mini_512.jsonl"
FROM_WEIGHT="pretrain"
FROM_RESUME=0
USE_WANDB=True
WANDB_PROJECT="minimind"

# Command to run the training script with the arguments
torchrun --nproc_per_node 8 train_full_sft.py \
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
  --use_wandb \
  --wandb_project $WANDB_PROJECT
