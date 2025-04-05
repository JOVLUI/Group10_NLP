import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from wordcloud import WordCloud
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# 下载必要的 NLTK 资源
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


def set_plot_style():
    """
    设置绘图样式的健壮方法
    """
    # 样式优先级列表
    style_options = [
        'seaborn-v0_8',
        'seaborn-darkgrid',
        'seaborn-whitegrid',
        'default'
    ]

    # 尝试设置样式
    for style in style_options:
        try:
            plt.style.use(style)
            print(f"成功使用样式: {style}")
            break
        except Exception as e:
            print(f"样式 {style} 设置失败: {e}")

    # 设置字体和主题
    sns.set(
        font_scale=1.2,  # 字体缩放
        rc={'figure.figsize': (12, 6)}  # 默认图形大小
    )


# 在类定义之前调用样式
set_plot_style()

class IMDB_EDA:
    def __init__(self, input_path):
        # 读取数据
        self.df = pd.read_csv(input_path)
        self.project_root = input_path.parent.parent
        self.eda_dir = self.project_root / 'eda'
        self.eda_dir.mkdir(exist_ok=True)

        # 预处理
        self.df['review_length'] = self.df['review'].apply(len)
        self.stop_words = set(stopwords.words('english'))

    def simple_clean(self, text):
        """文本清理函数  """
        text = re.sub(r'<.*?>', '', text)  # 移除 HTML 标签
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        return ' '.join(tokens)

    def basic_info(self):
        """基本数据信息"""
        print("=" * 50 + "\n1. 数据基本信息\n" + "=" * 50)
        print("数据概览:")
        print(self.df.info())
        print("\n列名:", self.df.columns.tolist())
        print("\n前 5 行数据:")
        print(self.df.head())
        print("\n缺失值统计:")
        print(self.df.isnull().sum())

    def sentiment_analysis(self):
        """情感分布分析"""
        print("\n" + "=" * 50 + "\n2. Sentiment 分布分析\n" + "=" * 50)

        # 计数分析
        sentiment_counts = self.df['sentiment'].value_counts()
        print("Sentiment 计数:")
        print(sentiment_counts)

        # 可视化
        plt.figure(figsize=(10, 6))

        # 使用不同颜色
        colors = ['#5784d6', '#d65b57']
        sentiment_counts.plot(kind='bar', color=colors)

        plt.title('Sentiment Distribution', fontsize=15)
        plt.xlabel('Sentiment (0 = Negative, 1 = Positive)', fontsize=12)
        plt.ylabel('Count', fontsize=12)

        # 添加数值标签
        for i, v in enumerate(sentiment_counts):
            plt.text(i, v, str(v), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.eda_dir / 'sentiment_distribution.png')
        plt.show()

    def review_length_analysis(self):
        """评论长度分析"""
        print("\n" + "=" * 50 + "\n3. Review 长度分析\n" + "=" * 50)

        # 长度统计
        print("Review 长度统计:")
        print(self.df['review_length'].describe())

        # 图1：箱线图
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='sentiment', y='review_length', data=self.df)
        plt.title('Review Length Distribution by Sentiment (Boxplot)', fontsize=12)
        plt.xlabel('Sentiment', fontsize=10)
        plt.ylabel('Review Length', fontsize=10)
        plt.tight_layout()
        plt.savefig(self.eda_dir / 'review_length_boxplot.png')
        plt.show()  # 添加 show() 方法
        plt.close()

        # 图2：直方图
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.df, x='review_length', hue='sentiment',
                     multiple='stack', bins=50)
        plt.title('Review Length Distribution by Sentiment (Histogram)', fontsize=12)
        plt.xlabel('Review Length', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.tight_layout()
        plt.savefig(self.eda_dir / 'review_length_histogram.png')
        plt.show()  # 添加 show() 方法
        plt.close()

        # 额外的长度分析
        print("\n长度详细分析:")
        print("按情感分组的长度统计:")
        print(self.df.groupby('sentiment')['review_length'].describe())

    def word_cloud_analysis(self):
        """词云分析"""
        print("\n" + "=" * 50 + "\n4. 词云分析\n" + "=" * 50)

        for sentiment in [0, 1]:
            reviews = self.df[self.df['sentiment'] == sentiment]['review']
            cleaned_text = [self.simple_clean(review) for review in
                            tqdm(reviews, desc=f"清洗 Sentiment {sentiment} 文本")]
            text = ' '.join(cleaned_text)

            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100
            ).generate(text)

            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud for Sentiment = {sentiment}')
            plt.tight_layout()
            plt.savefig(self.eda_dir / f'wordcloud_sentiment_{sentiment}.png')
            plt.show()
            plt.close()

    from tqdm import tqdm

    def top_words_analysis(self):
        """常用词分析"""
        print("\n" + "=" * 50 + "\n5. 最常见词汇\n" + "=" * 50)

        # 使用 tqdm 显示清理进度
        print("正在清理文本...")
        cleaned_words = []
        for review in tqdm(self.df['review'], desc="清理文本", unit="review"):
            cleaned_words.extend(self.simple_clean(review).split())

        # 统计词频
        print("\n正在统计词频...")
        word_freq = Counter()
        for word in tqdm(cleaned_words, desc="统计词频", unit="word"):
            word_freq[word] += 1

        # 获取前15个最常见词
        top_words = word_freq.most_common(15)

        # 创建 DataFrame 用于可视化
        df_word_freq = pd.DataFrame(top_words, columns=['word', 'frequency'])

        # 排序以便图表更易读
        df_word_freq = df_word_freq.sort_values('frequency')

        # 创建渐变色调色板
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_word_freq)))

        # 水平条形图
        plt.figure(figsize=(12, 8))

        # 使用渐变色
        bars = plt.barh(df_word_freq['word'], df_word_freq['frequency'], color=colors)

        plt.title('Top 15 Most Common Words', fontsize=15)
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Words', fontsize=12)

        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height() / 2,
                     f'{width:.0f}',
                     ha='left', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.eda_dir / 'top_words.png')
        plt.show()
        plt.close()


        # 打印词频
        print("\nTop 15 常见词汇:")
        for word, freq in top_words:
            print(f"{word}: {freq}")

    def run_eda(self):
        """执行全部 EDA 分析"""
        self.basic_info()
        self.sentiment_analysis()
        self.review_length_analysis()
        self.word_cloud_analysis()
        self.top_words_analysis()
        print("\nEDA 完成，图表已保存到 'eda' 文件夹")


# 主执行
if __name__ == "__main__":
    current_file = Path(__file__)
    project_root = current_file.parent.parent
    input_path = project_root / 'data' / 'cleaned_IMDB_Dataset.csv'

    eda = IMDB_EDA(input_path)
    eda.run_eda()
