import os

# 新旧标签映射表（原始标签 -> 新标签）
mapping = {
    # 纹样类
    "B-通用纹样": "B-纹样", "I-通用纹样": "I-纹样",
    "B-点缀纹样": "B-纹样", "I-点缀纹样": "I-纹样",
    "B-适合纹样": "B-纹样", "I-适合纹样": "I-纹样",

    # 颜色类
    "B-青色类": "B-颜色类", "I-青色类": "I-颜色类",
    "B-绿色类": "B-颜色类", "I-绿色类": "I-颜色类",
    "B-红色类": "B-颜色类", "I-红色类": "I-颜色类",
    "B-紫色类": "B-颜色类", "I-紫色类": "I-颜色类",
    "B-黑色类": "B-颜色类", "I-黑色类": "I-颜色类",
    "B-白色类": "B-颜色类", "I-白色类": "I-颜色类",
    "B-黄色类": "B-颜色类", "I-黄色类": "I-颜色类",
    "B-金色类": "B-颜色类", "I-金色类": "I-颜色类",

    # 工艺类
    "B-基层处理工艺": "B-工艺类", "I-基层处理工艺": "I-工艺类",
    "B-颜料制备工艺": "B-工艺类", "I-颜料制备工艺": "I-工艺类",
    "B-起稿与勾勒工艺": "B-工艺类", "I-起稿与勾勒工艺": "I-工艺类",
    "B-着色与渲染工艺": "B-工艺类", "I-着色与渲染工艺": "I-工艺类",
    "B-纹样绘制工艺": "B-工艺类", "I-纹样绘制工艺": "I-工艺类",
    "B-贴金与装饰工艺": "B-工艺类", "I-贴金与装饰工艺": "I-工艺类",

    # 工具类
    "B-绘制工具": "B-工具类", "I-绘制工具": "I-工具类",
    "B-加工工具": "B-工具类", "I-加工工具": "I-工具类",

    # 保留不变的部分
    "B-制度术语": "B-制度术语", "I-制度术语": "I-制度术语",
    "B-建筑载体": "B-建筑载体", "I-建筑载体": "I-建筑载体",
    "B-矿物颜料": "B-矿物颜料", "I-矿物颜料": "I-矿物颜料",
    "B-植物颜料": "B-植物颜料", "I-植物颜料": "I-植物颜料",
    "B-加工辅料": "B-加工辅料", "I-加工辅料": "I-加工辅料",
    "B-规则/约束": "B-规则/约束", "I-规则/约束": "I-规则/约束",
    "O": "O"
}

def convert_file(input_path, output_path=None):
    if output_path is None:
        output_path = input_path
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            new_lines.append("")  # 保留空行
        elif " " in line:
            # 标签行
            labels = line.split()
            new_labels = [mapping.get(l, "O") for l in labels]
            new_lines.append(" ".join(new_labels))
        else:
            # 文本行
            new_lines.append(line)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))
    print(f"转换完成: {output_path}")

if __name__ == "__main__":
    data_dir = r"C:\Users\59340\Desktop\yzfs_ner\data"
    train_path = os.path.join(data_dir, "train_bio.txt")
    test_path = os.path.join(data_dir, "test_bio.txt")

    # 备份原文件（可选）
    # import shutil
    # shutil.copy(train_path, train_path + ".bak")
    # shutil.copy(test_path, test_path + ".bak")

    convert_file(train_path)
    convert_file(test_path)