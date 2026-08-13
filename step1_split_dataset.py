import json
import random

input_file = "cimo_train_dataset.jsonl"
train_file = "train.jsonl"
test_file = "test.jsonl"

def split_dataset():
    # 1. 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data =[json.loads(line.strip()) for line in f if line.strip()]
    
    # 2. 随机打乱
    random.seed(42)
    random.shuffle(data)
    
    # 3. 划分 90% 训练, 10% 测试
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # 4. 保存
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"✅ 数据集划分完成！")
    print(f"总数据量: {len(data)}")
    print(f"训练集: {len(train_data)} 条 -> {train_file}")
    print(f"测试集: {len(test_data)} 条 -> {test_file}")

if __name__ == "__main__":
    split_dataset()