"""
用预训练的Bert模型微调IMDB数据集。
IMDB数据集的1是positive，0是negative。
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


data = load_dataset("./imdb")
print(data)