import torch
import re
from transformers import BertTokenizerFast, BertForTokenClassification

# 配置
MODEL_PATH = r"C:\Users\59340\Desktop\yzfs_ner\models\yzfs_ner_model"   # 使用训练好的模型
ORIGIN_FILE = r"C:\Users\59340\Desktop\yzfs_ner\data\原文.txt"
OUTPUT_FILE = r"C:\Users\59340\Desktop\yzfs_ner\data\原文_实体标注_逐句.txt"

# 加载模型和 tokenizer
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForTokenClassification.from_pretrained(MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 标签映射
id2label = model.config.id2label

def predict(text):
    """对单个句子进行预测，返回字符级标签列表"""
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=256,
        padding="max_length",
        return_offsets_mapping=True
    )
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=2)[0].tolist()
    aligned = []
    prev_char = -1
    for (start, end), pred in zip(offset_mapping, preds):
        if start == 0 and end == 0:
            continue
        if start != prev_char:
            aligned.append((text[start], id2label[pred]))
            prev_char = start
    return aligned

def merge_entities(aligned):
    """将字符级标签合并为实体列表"""
    entities = []
    current_entity = ""
    current_type = ""
    for char, label in aligned:
        if label.startswith("B-"):
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = char
            current_type = label[2:]
        elif label.startswith("I-") and current_type == label[2:]:
            current_entity += char
        else:
            if current_entity:
                entities.append((current_entity, current_type))
                current_entity = ""
                current_type = ""
    if current_entity:
        entities.append((current_entity, current_type))
    return entities

def split_sentences(text):
    """按句号、问号、感叹号、分号等切分句子（保留标点）"""
    # 使用正则匹配句子边界，保留标点
    sentences = re.split(r'([。！？；])', text)
    # 将标点与前面的内容合并
    result = []
    for i in range(0, len(sentences)-1, 2):
        sent = sentences[i] + sentences[i+1]
        if sent.strip():
            result.append(sent.strip())
    # 处理可能剩余的无标点部分
    if len(sentences) % 2 == 1:
        last = sentences[-1].strip()
        if last:
            result.append(last)
    return result

# 读取原文
with open(ORIGIN_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# 按句子切分
sentences = split_sentences(content)

# 写入输出文件
with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    for sent in sentences:
        # 跳过过短或无意义的句子
        if len(sent) < 2:
            continue
        aligned = predict(sent)
        entities = merge_entities(aligned)
        f_out.write(f"【句子】\n{sent}\n")
        f_out.write("【实体】\n")
        if entities:
            for ent, typ in entities:
                f_out.write(f"{ent}\t{typ}\n")
        else:
            f_out.write("(无)\n")
        f_out.write("\n" + "-"*50 + "\n")

print(f"结果已保存至 {OUTPUT_FILE}")