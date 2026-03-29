import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from seqeval.metrics import classification_report

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Users\59340\Desktop\yzfs_ner\models\yzfs_ner_model"
TEST_FILE = r"C:\Users\59340\Desktop\yzfs_ner\data\test_bio.txt"

# 加载模型和 tokenizer
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForTokenClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# 从模型配置中获取标签映射（推荐）
id2label = model.config.id2label
label2id = model.config.label2id

# 加载测试集（同前）
def load_test_data(file_path):
    sentences = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        blocks = f.read().split("\n\n")
        for block in blocks:
            if not block.strip():
                continue
            sent, label = block.strip().split("\n")
            sentences.append(sent.strip())
            labels.append(label.strip().split())
    return sentences, labels

test_sents, test_labels = load_test_data(TEST_FILE)

def predict_for_eval(text, true_labels):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
        return_offsets_mapping=True
    )
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=2)[0].tolist()

    pred_labels = []
    prev_char_idx = -1
    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:
            continue
        if start != prev_char_idx:
            pred_labels.append(id2label[preds[i]])
            prev_char_idx = start
    # 截断/补齐
    pred_labels = pred_labels[:len(true_labels)] + ["O"] * (len(true_labels) - len(pred_labels))
    return pred_labels

all_preds = []
all_trues = []
for sent, true in zip(test_sents, test_labels):
    pred = predict_for_eval(sent, true)
    all_preds.append(pred)
    all_trues.append(true)

print("=" * 60)
print("模型在测试集上的实体级别评估结果（合并后）")
print("=" * 60)
print(classification_report(all_trues, all_preds, digits=4))