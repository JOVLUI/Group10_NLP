from pathlib import Path
import pandas as pd

# 获取当前文件的路径
current_file = Path(__file__)
# 获取项目根目录
project_root = current_file.parent.parent
# 构建数据集路径
filepath = project_root / 'data' / 'IMDB_Dataset.csv'

# pands读取数据并打印前几行
df = pd.read_csv(filepath, encoding='latin-1')  # Latin-1 编码
print("=" * 100)
print(df.head())
print("=" * 100)
print("\n")

# 打印结果是[5 rows x 14 columns]，继续打印前5行内容查看原因
with open(filepath, 'r', encoding='latin-1') as f:
    for i, line in enumerate(f):
        if i < 5:  # 打印前 5 行
            print(line.strip())
print("=" * 100)
print("\n")

# 每一行后面多了12个逗号，需要手动指定前两列
df = pd.read_csv(filepath, encoding='latin-1', names=['review', 'sentiment'], header=0, usecols=[0, 1])
print(df.head())
print(f"列数: {len(df.columns)}")
print("=" * 100)
print("\n")

# 处理缺失值
df.dropna(inplace=True)

# 转换 sentiment 列为数值
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 打印处理后的数据详细信息
print("=" * 50)
print("数据清洗后的详细信息")
print("=" * 50)
print("\n列名:", df.columns.tolist())  # 显示所有列名
print("\n前 5 行数据:")
print(df.head())  # 显示前 5 行
print("\n数据统计描述:")
print(df.describe())  # 显示数值列的统计信息（如 sentiment 的均值、计数等）
print("=" * 50)

# 保存清洗后的数据
output_path = project_root / 'data' / 'cleaned_IMDB_Dataset.csv'
df.to_csv(output_path, index=False)
print("数据准备完成，已保存到:", output_path)