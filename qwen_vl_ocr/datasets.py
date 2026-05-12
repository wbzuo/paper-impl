import os
import json

# 读取train.json
abs_folder = r"C:\Users\Administrator\Desktop\251\项目\train_20200121\resume_train_20200121"
json_file = os.path.join(abs_folder, "train_data.json")
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)


import json
# 打开文件并读取内容

conversations = []

with open(json_file, 'r', encoding='utf-8') as file:
    # 使用json.load()方法解析JSON数据 
    data = json.load(file)
    # 打印解析后的Python对象
    for i, file_id in enumerate(data):
        image_path = os.path.join(abs_folder, f"images\\{ file_id }.png")
        # print(i, file_id, image_path, data[file_id])
        conversations.append({
            "id": f"identity_{i+1}",
            "conversations": [
                {
                    "role": "user",
                    "value": f"{image_path}"
                },
                {
                    "role": "assistant", 
                    "value": str(data[file_id])
                }
            ]
        })

# 80% 使用训练集 20% 使用测试集
train_conversations = conversations[:1600]
val_conversations = conversations[1600:]

train_json_path = os.path.join(current_dir, "train_json.json")
val_json_path = os.path.join(current_dir, "val_json.json")
# Save train set
with open(train_json_path, 'w', encoding='utf-8') as f:
    json.dump(train_conversations, f, ensure_ascii=False, indent=2)

# Save validation set 
with open(val_json_path, 'w', encoding='utf-8') as f:
    json.dump(val_conversations, f, ensure_ascii=False, indent=2)    
