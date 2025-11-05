#!/bin/bash

# MiniMind 预训练脚本
# 使用方法: ./run_pretrain.sh [配置名称]

# 设置默认配置
CONFIG=${1:-"base"}

echo "启动 MiniMind 预训练 - 配置: $CONFIG"
echo "训练脚本: ./trainer/train_pretrain.py"

# 根据配置设置不同参数
case $CONFIG in
    "base")
        # 基础配置 - 快速验证
        BATCH_SIZE=32
        ACCUMULATION_STEPS=8
        EPOCHS=1
        HIDDEN_SIZE=512
        NUM_LAYERS=8
        SAVE_INTERVAL=100
        ;;
    "medium")
        # 中等配置 - 平衡性能
        BATCH_SIZE=64
        ACCUMULATION_STEPS=4
        EPOCHS=2
        HIDDEN_SIZE=768
        NUM_LAYERS=12
        SAVE_INTERVAL=500
        ;;
    "large")
        # 大型配置 - 充分训练
        BATCH_SIZE=128
        ACCUMULATION_STEPS=2
        EPOCHS=6
        HIDDEN_SIZE=1024
        NUM_LAYERS=16
        SAVE_INTERVAL=1000
        ;;
    "test")
        # 测试配置 - 快速调试
        BATCH_SIZE=8
        ACCUMULATION_STEPS=2
        EPOCHS=1
        HIDDEN_SIZE=256
        NUM_LAYERS=4
        SAVE_INTERVAL=50
        ;;
    *)
        echo "未知配置: $CONFIG，使用基础配置"
        BATCH_SIZE=32
        ACCUMULATION_STEPS=8
        EPOCHS=1
        HIDDEN_SIZE=512
        NUM_LAYERS=8
        SAVE_INTERVAL=100
        ;;
esac

echo "训练配置:"
echo "  Batch Size: $BATCH_SIZE"
echo "  累积步数: $ACCUMULATION_STEPS"
echo "  训练轮数: $EPOCHS"
echo "  隐藏层维度: $HIDDEN_SIZE"
echo "  隐藏层数量: $NUM_LAYERS"
echo "  保存间隔: $SAVE_INTERVAL"

# 执行训练命令
python ./trainer/train_pretrain.py \
    --save_dir "../out" \
    --save_weight "pretrain_${CONFIG}" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 5e-4 \
    --device "cuda:0" \
    --dtype "bfloat16" \
    --num_workers 4 \
    --accumulation_steps $ACCUMULATION_STEPS \
    --grad_clip 1.0 \
    --log_interval 100 \
    --save_interval $SAVE_INTERVAL \
    --hidden_size $HIDDEN_SIZE \
    --num_hidden_layers $NUM_LAYERS \
    --max_seq_len 512 \
    --use_moe 0 \
    --data_path "../dataset/pretrain_hq.jsonl" \
    --from_weight "none" \
    --from_resume 0 \
    --use_wandb \
    --wandb_project "MiniMind-Pretrain"

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "训练完成!"
else
    echo "训练失败!"
    exit 1
fi