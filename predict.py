import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# 模型路径（和你训练时一致）
model_path = r'C:\Users\59340\Desktop\yzfs_ner\models\yzfs_ner_model'

# 必须用 Fast Tokenizer（和训练时保持一致）
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForTokenClassification.from_pretrained(model_path)

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 标签映射（和训练时完全一致：颜色合并为颜色类、工艺合并为工艺类、纹样保留细分）
# ID2LABEL = {
#     0: "O",
#     # 制度/建筑类
#     1: "B-制度术语", 2: "I-制度术语",
#     3: "B-建筑载体", 4: "I-建筑载体",
#     # 纹样类（保留细分）
#     5: "B-通用纹样", 6: "I-通用纹样",
#     7: "B-点缀纹样", 8: "I-点缀纹样",
#     9: "B-适合纹样", 10: "I-适合纹样",
#     # 合并后的颜色类（替代原来的青色/绿色/红色等细分）
#     11: "B-颜色类", 12: "I-颜色类",
#     # 颜料/辅料类
#     13: "B-矿物颜料", 14: "I-矿物颜料",
#     15: "B-植物颜料", 16: "I-植物颜料",
#     17: "B-加工辅料", 18: "I-加工辅料",
#     # 合并后的工艺类（替代原来的基层处理/颜料制备等细分）
#     19: "B-工艺类", 20: "I-工艺类",
#     # 工具/规则类
#     21: "B-绘制工具", 22: "I-绘制工具",
#     23: "B-加工工具", 24: "I-加工工具",
#     25: "B-规则/约束", 26: "I-规则/约束"
# }
ID2LABEL = {
    0: "O",
    1: "B-制度术语", 2: "I-制度术语",
    3: "B-建筑载体", 4: "I-建筑载体",
    5: "B-纹样", 6: "I-纹样",
    7: "B-颜色类", 8: "I-颜色类",
    9: "B-矿物颜料", 10: "I-矿物颜料",
    11: "B-植物颜料", 12: "I-植物颜料",
    13: "B-加工辅料", 14: "I-加工辅料",
    15: "B-工艺类", 16: "I-工艺类",
    17: "B-工具类", 18: "I-工具类",
    19: "B-规则/约束", 20: "I-规则/约束"
}

def predict(text):
    # 1. 编码文本，拿到 offset_mapping
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding="max_length",
        return_offsets_mapping=True
    )

    # 2. ✅ 关键：把 offset_mapping 单独拿出来，不传给模型
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    # 剩下的 inputs 只保留模型需要的字段
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)  # 现在不会报错了

    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    labels = [ID2LABEL[p.item()] for p in predictions[0]]

    # 3. 对齐到原句（根据字符偏移）
    aligned_result = []
    prev_char_idx = -1
    for token, label, (start, end) in zip(tokens, labels, offset_mapping):
        if start == 0 and end == 0:
            continue  # 跳过 [CLS]、[SEP]、[PAD]
        if start != prev_char_idx:
            aligned_result.append((text[start], label))
            prev_char_idx = start
    return aligned_result


# ===================== 批量验证文本 =====================
if __name__ == '__main__':
    # 先测试训练集原文
    train_text = "凡五遍之制，用于特定等级之建筑。"
    print("【测试训练集原文】", train_text)
    result = predict(train_text)
    print("原始识别结果:")
    for c, l in result:
        print(f"  {c} → {l}")

    test_texts = [
        "凡青绿棱间装，柱身内筍文，柱头作四合青绿退晕如意头。",
        "飞子正面作合晕，两旁并退晕，或素绿，仰版素红。",
        "大连檐立面作三角叠晕柿蒂华，或作霞光。",
        "解绿结华装：枓栱缘内朱地上间诸华，外留青绿叠晕缘道。",
        "丹粉刷饰：材木面上用土朱通刷，下棱用白粉阑界缘道。",
        "凡画松文，身内通刷土黄，先以墨笔界画，次以紫檀间刷。",
        "雌黄调制：先捣次研，用热汤淘细华，澄去清水入胶水。",
        "杂间装之制：五彩遍装六分，碾玉装四分，相间品配。",
        "凡用桐油，先以文武火煎令清，次下松脂搅候化。",
        "彩画作上等：五彩装饰、青绿碾玉，间用金同。"
    ]

    print("=" * 60)
    print("          《营造法式》彩画作 NER 模型验证结果")
    print("=" * 60)

    for idx, text in enumerate(test_texts, 1):
        print(f"\n【第 {idx} 句】{text}")
        print("-" * 50)
        result = predict(text)

        # 打印原始识别结果（每个字的标签）
        print("【原始识别】")
        for c, l in result:
            print(f"  {c} → {l}")

        # 自动合并实体并输出
        print("\n【识别到的实体】")
        entity = ""
        etype = ""
        for char, label in result:
            if label.startswith("B-"):
                if entity:
                    print(f"▶ {entity} → {etype}")
                entity = char
                etype = label.split("-")[1]
            elif label.startswith("I-") and etype == label.split("-")[1]:
                entity += char
            else:
                if entity:
                    print(f"▶ {entity} → {etype}")
                    entity = ""
                    etype = ""
        if entity:
            print(f"▶ {entity} → {etype}")

    print("\n" + "=" * 60)