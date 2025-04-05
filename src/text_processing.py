import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import time
from pathlib import Path


class TextPreprocessor:
    def __init__(self):
        """初始化文本预处理器"""
        # 下载必要的 NLTK 资源
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

        # 初始化停用词和词形还原器
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # 启用tqdm的pandas集成（仅用于分词环节）
        tqdm.pandas()

    def clean_text(self, text):
        """
        文本清理的核心方法（无进度条）
        """
        # 转换为小写
        text = text.lower()

        # 删除 HTML 标签
        text = re.sub(r'<.*?>', '', text)

        # 删除特殊字符和数字
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        return text.strip()

    def tokenize(self, text):
        """
        分词和预处理（无进度条）
        """
        # 实际分词处理
        tokens = word_tokenize(text)

        # 删除停用词并进行词形还原
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        return tokens

    def preprocess_dataframe(self, df):
        """
        处理整个数据集
        """
        print("已执行文本清理...")
        print("=" * 50)
        # 文本清理
        df['cleaned_review'] = df['review'].apply(self.clean_text)

        print("\n开始进行分词处理...")
        # 分词处理（保留进度条）
        df['processed_review'] = df['cleaned_review'].progress_apply(
            lambda x: ' '.join(self.tokenize(x))
        )

        return df

    def vectorize_text(self, df, max_features=5000):
        """
        文本向量化
        """
        if df['sentiment'].isna().any():
            nan_count = df['sentiment'].isna().sum()
            print(f"\n警告：发现 {nan_count} 条记录的sentiment为空值，已自动删除")
            df = df.dropna(subset=['sentiment'])

        print("\n正在进行TF-IDF文本向量化...")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english'
        )

        X = vectorizer.fit_transform(df['processed_review'])
        y = df['sentiment']

        return X, y, vectorizer

    def process_pipeline(self, input_path, output_path=None, max_features=5000):
        """
        完整处理流程
        """
        print("读取数据集...")
        df = pd.read_csv(input_path)

        # 文本预处理
        df = self.preprocess_dataframe(df)

        # 文本向量化
        X, y, vectorizer = self.vectorize_text(df, max_features)

        # 数据集划分
        print("\n划分训练集和测试集...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        if output_path:
            print("\n保存处理后的数据...")
            np.savez(
                output_path,
                X_train=X_train.toarray(),  # 将稀疏矩阵转换为普通数组
                X_test=X_test.toarray(),  # 将稀疏矩阵转换为普通数组
                y_train=y_train.to_numpy(),  # 如果是pandas Series，转换为numpy数组
                y_test=y_test.to_numpy(),  # 如果是pandas Series，转换为numpy数组
                allow_pickle=True
            )

        return X_train, X_test, y_train, y_test, vectorizer


if __name__ == "__main__":
    # 获取项目根目录
    project_root = Path(__file__).parent.parent

    # 构建路径
    input_path = project_root / 'data' / 'cleaned_IMDB_Dataset.csv'
    output_path = project_root / 'data' / 'processed_data.npz'

    # 初始化处理器
    preprocessor = TextPreprocessor()

    # 执行完整处理流程
    print("开始文本处理流程...")
    start_time = time.time()

    try:
        X_train, X_test, y_train, y_test, vectorizer = preprocessor.process_pipeline(
            input_path,
            output_path,
            max_features=3000
        )

        # 保存向量化器
        import joblib

        joblib.dump(vectorizer, project_root / 'models' / 'tfidf_vectorizer.joblib')

        print("\n处理完成！耗时: %.2f秒" % (time.time() - start_time))
        print("训练集大小:", X_train.shape)
        print("测试集大小:", X_test.shape)
        print("向量化器已保存")
    except Exception as e:
        print(f"\n处理过程中发生错误: {str(e)}")
        import traceback

        traceback.print_exc()