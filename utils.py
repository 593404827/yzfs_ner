import os
from torch.utils.data import Dataset

def load_bio_data(file_path):
    """加载BIO格式标注数据"""
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        label = []
        for line in f:
            line = line.strip()
            if line.startswith('# TEXT'):
                continue
            if not line:
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
            else:
                word, tag = line.split()
                sentence.append(word)
                label.append(tag)
        if sentence:
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels

# 标签列表（和你的标注完全对应）
LABEL_LIST = [
    'O',
    'B-制度术语', 'I-制度术语',
    'B-建筑载体', 'I-建筑载体',
    'B-着色与渲染工艺', 'I-着色与渲染工艺',
    'B-加工辅料', 'I-加工辅料',
    'B-白色类', 'I-白色类',
    'B-青色类', 'I-青色类',
    'B-红色类', 'I-红色类',
    'B-黄色类', 'I-黄色类',
    'B-金色类', 'I-金色类',
    'B-绿色类', 'I-绿色类',
    'B-紫色类', 'I-紫色类',
    'B-黑色类', 'I-黑色类',
    'B-矿物颜料', 'I-矿物颜料',
    'B-植物颜料', 'I-植物颜料',
    'B-通用纹样', 'I-通用纹样',
    'B-点缀纹样', 'I-点缀纹样',
    'B-适合纹样', 'I-适合纹样',
    'B-基层处理工艺', 'I-基层处理工艺',
    'B-颜料制备工艺', 'I-颜料制备工艺',
    'B-起稿与勾勒工艺', 'I-起稿与勾勒工艺',
    'B-贴金与装饰工艺', 'I-贴金与装饰工艺',
    'B-纹样绘制工艺', 'I-纹样绘制工艺',
    'B-绘制工具', 'I-绘制工具',
    'B-加工工具', 'I-加工工具',
    'B-规则/约束', 'I-规则/约束'
]

LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}