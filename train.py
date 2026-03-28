import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.optim import AdamW
from tqdm import tqdm
import os
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------------------- 极速参数 ----------------------
MAX_LEN = 256
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
NUM_EPOCHS = 15
PATIENCE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = r"C:\Users\59340\Desktop\yzfs_ner\data"
TRAIN_FILE = os.path.join(DATA_PATH, "train_bio.txt")
TEST_FILE = os.path.join(DATA_PATH, "test_bio.txt")
MODEL_SAVE_PATH = r"C:\Users\59340\Desktop\yzfs_ner\models\yzfs_ner_model"

# 标签列表（和你的数据完全匹配，无需修改）
LABEL_LIST = [
    "O",
    "B-制度术语", "I-制度术语",
    "B-建筑载体", "I-建筑载体",
    "B-通用纹样", "I-通用纹样",
    "B-点缀纹样", "I-点缀纹样",
    "B-适合纹样", "I-适合纹样",
    "B-青色类", "I-青色类",
    "B-绿色类", "I-绿色类",
    "B-红色类", "I-红色类",
    "B-紫色类", "I-紫色类",
    "B-黑色类", "I-黑色类",
    "B-白色类", "I-白色类",
    "B-黄色类", "I-黄色类",
    "B-金色类", "I-金色类",
    "B-矿物颜料", "I-矿物颜料",
    "B-植物颜料", "I-植物颜料",
    "B-加工辅料", "I-加工辅料",
    "B-基层处理工艺", "I-基层处理工艺",
    "B-颜料制备工艺", "I-颜料制备工艺",
    "B-起稿与勾勒工艺", "I-起稿与勾勒工艺",
    "B-着色与渲染工艺", "I-着色与渲染工艺",
    "B-纹样绘制工艺", "I-纹样绘制工艺",
    "B-贴金与装饰工艺", "I-贴金与装饰工艺",
    "B-绘制工具", "I-绘制工具",
    "B-加工工具", "I-加工工具",
    "B-规则/约束", "I-规则/约束"
]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# ---------------------- 数据集（核心修复：适配整行文本+整行标签格式） ----------------------
class YZFS_NER_Dataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sentences, self.labels = self.load_data(file_path)

    def load_data(self, file_path):
        sentences = []
        labels = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]  # 过滤空行
            # 你的数据是：文本行 + 标签行 交替出现，按索引两两配对
            for i in range(0, len(lines), 2):
                sent_line = lines[i]    # 第1行：原始文本
                label_line = lines[i+1] # 第2行：对应BIO标签（空格分隔）
                # 转成字符列表和标签列表，保证一一对应
                sent_chars = list(sent_line)
                label_list = label_line.split()
                # 严格保证字符数和标签数一致（过滤异常行）
                if len(sent_chars) == len(label_list):
                    sentences.append(sent_chars)
                    labels.append(label_list)
        print(f"加载数据完成，有效样本数：{len(sentences)}")
        return sentences, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent_chars = self.sentences[idx]  # 字符列表
        label_list = self.labels[idx]    # 标签列表
        sent_str = "".join(sent_chars)   # 转回字符串用于分词

        # BERT分词：保留原始字符映射，关闭自动填充（后续统一处理）
        encoding = self.tokenizer(
            sent_str,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt",
            return_offsets_mapping=False
        )

        label_ids = []
        word_ids = encoding.word_ids(batch_index=0)  # 获取token对应原始字符的索引
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # [CLS]/[SEP]/PAD 特殊符号，标签设为-100（模型自动忽略）
                label_ids.append(-100)
            else:
                # 防越界：word_idx超出标签长度则设为-100
                if word_idx >= len(label_list):
                    label_ids.append(-100)
                # 同一个原始字符的多个子词，复用同一个标签（NER标准做法）
                elif word_idx != previous_word_idx:
                    label_ids.append(LABEL2ID[label_list[word_idx]])
                else:
                    label_ids.append(LABEL2ID[label_list[word_idx]])
            previous_word_idx = word_idx

        # 统一填充/截断到MAX_LEN，标签填充-100
        if len(label_ids) < self.max_len:
            label_ids += [-100] * (self.max_len - len(label_ids))
        else:
            label_ids = label_ids[:self.max_len]
        # 对input_ids和attention_mask做同样的填充，保证长度一致
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()
        if len(input_ids) < self.max_len:
            pad_len = self.max_len - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
        else:
            input_ids = input_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }

# ---------------------- 模型加载 ----------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
model = BertForTokenClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=len(LABEL_LIST),
    id2label=ID2LABEL,
    label2id=LABEL2ID
).to(DEVICE)

# 加载数据集
train_dataset = YZFS_NER_Dataset(TRAIN_FILE, tokenizer, MAX_LEN)
test_dataset = YZFS_NER_Dataset(TEST_FILE, tokenizer, MAX_LEN)

# 构建DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 优化器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# ---------------------- 训练函数 ----------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="Training", leave=False)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        # 前向传播：模型自动计算损失（忽略label=-100的位置）
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        # 反向传播+更新参数
        loss.backward()
        optimizer.step()
        # 更新进度条
        loop.set_postfix(loss=loss.item(), avg_loss=total_loss/(loop.n+1))
    return total_loss / len(loader)

# ---------------------- 评估函数（过滤-100，精准计算指标） ----------------------
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        loop = tqdm(loader, desc="Evaluating", leave=False)
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            # 获取预测结果
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            # 核心：只保留标签≠-100的有效位置，过滤padding/特殊符号
            active_mask = (labels != -100) & (attention_mask == 1)
            active_preds = preds[active_mask]
            active_labels = labels[active_mask]

            # 收集结果
            all_preds.extend(active_preds.cpu().numpy())
            all_labels.extend(active_labels.cpu().numpy())
            loop.set_postfix(loss=outputs.loss.item())

    # 计算评估指标，zero_division=0避免除0报错（无样本时指标为0）
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds,
        average='macro',
        zero_division=0
    )
    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ---------------------- 开始训练（带早停） ----------------------
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

best_val_loss = float('inf')
early_stop_count = 0

print("="*50)
print("🚀 训练启动")
print(f"设备：{DEVICE}")
print(f"训练集：{len(train_dataset)} 样本 | 测试集：{len(test_dataset)} 样本")
print(f"Batch Size：{BATCH_SIZE} | 最大长度：{MAX_LEN} | 学习率：{LEARNING_RATE}")
print("="*50)

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    print(f"\n📌 Epoch {epoch + 1}/{NUM_EPOCHS}")

    # 训练
    train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
    # 验证
    val_metrics = eval_epoch(model, test_loader, DEVICE)

    # 计算耗时
    epoch_time = time.time() - start_time
    # 打印结果
    print(f"🎯 训练损失：{train_loss:.4f} | 验证损失：{val_metrics['loss']:.4f}")
    print(f"📊 准确率：{val_metrics['accuracy']:.4f} | 精确率：{val_metrics['precision']:.4f}")
    print(f"📈 召回率：{val_metrics['recall']:.4f} | F1值：{val_metrics['f1']:.4f}")
    print(f"⏱ 耗时：{epoch_time:.2f} 秒")

    # 早停逻辑：验证损失下降则保存模型
    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        early_stop_count = 0
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)
        print("✅ 模型已保存（验证损失最优）")
    else:
        early_stop_count += 1
        print(f"⚠️  验证损失未下降，早停计数：{early_stop_count}/{PATIENCE}")
        if early_stop_count >= PATIENCE:
            print("🛑 早停触发，训练结束！")
            break

print("\n🎉 训练全部完成！最优模型已保存至：", MODEL_SAVE_PATH)