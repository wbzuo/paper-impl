"""
用预训练的Bert模型微调IMDB数据集，并使用SwanLabCallback回调函数将结果上传到SwanLab。
IMDB数据集的1是positive，0是negative。
"""

import torch
import numpy as np
from datasets import load_dataset
from swanlab.integration.transformers import SwanLabCallback
import swanlab
from modelscope import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 定义预测函数
def predict(text, model, tokenizer, CLASS_NAME):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return int(predicted_class)

# 计算准确率的函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).astype(np.float32).mean()
    return {"accuracy": accuracy}

# 加载IMDB数据集
dataset = load_dataset('stanfordnlp/imdb')

# 加载预训练的BERT tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', num_labels=2)

# 数据预处理
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# 测试数据
test_reviews = [
    "I absolutely loved this movie! The storyline was captivating and the acting was top-notch.",
    "This movie was a complete waste of time. The plot was predictable.",
    "An excellent film with a heartwarming story.",
    "I found the movie to be quite boring.",
    "A masterpiece! The visuals were stunning.",
    "Terrible movie. The script was awful.",
    "A delightful film with humor and drama.",
    "I was very disappointed with this movie.",
    "One of the best movies I've seen this year.",
    "I didn't enjoy this movie at all."
]

CLASS_NAME = {0: "negative", 1: "positive"}
true_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# 设置swanlab回调函数
swanlab_callback = SwanLabCallback(
    project='text_classification',
    experiment_name='BERT-IMDB',
    config={
        'dataset': 'IMDB', 
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3
    }
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    callbacks=[swanlab_callback],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# 训练模型
print("开始训练...")
trainer.train()

# 最终评估
final_eval = trainer.evaluate()
final_accuracy = final_eval.get('eval_accuracy', 0)
print(f"最终验证集准确率: {final_accuracy:.4f}")

# 保存模型
model.save_pretrained('./sentiment_model')
tokenizer.save_pretrained('./sentiment_model')

# 测试预测
print("\n测试预测...")
model.eval()
model.to('cpu')

predictions = []
text_list = []

for i, review in enumerate(test_reviews):
    label = predict(review, model, tokenizer, CLASS_NAME)
    predictions.append(label)
    text_list.append(swanlab.Text(review, caption=f"{label}-{CLASS_NAME[label]}"))
    print(f"文本: {review[:50]}... -> 预测: {CLASS_NAME[label]}")

# 计算测试准确率
test_accuracy = (np.array(predictions) == np.array(true_labels)).mean()
print(f"测试集准确率: {test_accuracy:.4f}")

# 记录结果到SwanLab
swanlab.log({
    "final_accuracy": final_accuracy,
    "test_accuracy": test_accuracy,
    "predict": text_list
})

print("训练完成!")
swanlab.finish()