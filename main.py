import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import traceback
import os


class MultiModelSentimentPredictor:
    def __init__(self, models_dir, vectorizer_path):
        """
        初始化多模型情感预测器
        """
        # 下载必要的NLTK资源
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        # 加载模型和向量化器
        try:
            # 加载所有模型
            self.models = {}
            for model_file in os.listdir(models_dir):
                if model_file.endswith('.joblib') and 'model' in model_file:
                    model_name = model_file.split('_')[0]
                    model_path = os.path.join(models_dir, model_file)
                    self.models[model_name] = joblib.load(model_path)

            # 加载向量化器
            self.vectorizer = joblib.load(vectorizer_path)

            # 打印模型信息
            self.print_models_info()
        except Exception as e:
            print(f"模型加载错误: {e}")
            traceback.print_exc()
            raise

        # 停用词
        self.stop_words = set(stopwords.words('english'))

    def print_models_info(self):
        """
        打印模型和向量化器的诊断信息
        """
        print("模型诊断信息:")
        try:
            # 打印已加载模型
            print("已加载模型:")
            for name, model in self.models.items():
                print(f"- {name}: {type(model).__name__}")

            # 打印向量化器信息
            print(f"向量化器类型: {type(self.vectorizer).__name__}")

            # 打印特征名称
            feature_names = self.vectorizer.get_feature_names_out()
            print(f"特征数量: {len(feature_names)}")
            print("前10个特征:", feature_names[:10])
        except Exception as e:
            print(f"诊断信息获取失败: {e}")

    def preprocess_text(self, text):
        """
        文本预处理
        """
        # 转小写
        text = text.lower()

        # 去除特殊字符
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # 分词
        tokens = word_tokenize(text)

        # 去除停用词
        tokens = [token for token in tokens if token not in self.stop_words]

        return ' '.join(tokens)

    def predict_sentiment(self, text):
        """
        使用多个模型预测情感
        """
        try:
            # 文本预处理
            processed_text = self.preprocess_text(text)

            # 文本向量化
            text_vectorized = self.vectorizer.transform([processed_text])

            # 安全地处理向量化结果
            if text_vectorized.shape[1] == 0:
                raise ValueError("文本向量化后特征为空")

            # 存储每个模型的预测结果
            results = {}

            # 对每个模型进行预测
            for model_name, model in self.models.items():
                # 预测类别
                prediction = model.predict(text_vectorized)

                # 预测概率
                proba = model.predict_proba(text_vectorized)

                # 安全地获取类别和置信度
                pred_class = int(prediction[0])  # 确保转换为整数
                confidence = proba[0][pred_class]

                # 转换为可读结果
                sentiment = "正面" if pred_class == 1 else "负面"

                # 存储结果
                results[model_name] = {
                    'sentiment': sentiment,
                    'confidence': confidence
                }

            return results

        except Exception as e:
            print(f"预测过程详细错误: {e}")
            traceback.print_exc()
            raise


def main():
    print("🎬 多模型电影评论情感分析 - 控制台测试")
    print("=" * 50)

    try:
        # 初始化多模型预测器
        predictor = MultiModelSentimentPredictor(
            models_dir='models',  # 包含所有模型的目录
            vectorizer_path='models/tfidf_vectorizer.joblib'
        )

        # 测试模式选择
        while True:
            print("\n选择测试模式:")
            print("1. 手动输入测试")
            print("2. 预设评论测试")
            print("3. 退出")

            choice = input("请输入选项(1/2/3): ").strip()

            if choice == '1':
                # 手动输入测试
                user_input = input("\n请输入电影评论: ")
                if user_input.strip():
                    try:
                        results = predictor.predict_sentiment(user_input)
                        print(f"\n📊 分析结果:")
                        for model_name, result in results.items():
                            print(f"{model_name}模型:")
                            print(f"  情感倾向: {result['sentiment']}")
                            print(f"  置信度: {result['confidence']:.2%}")
                    except Exception as e:
                        print(f"预测出错: {e}")
                else:
                    print("输入不能为空!")

            elif choice == '2':
                # 预设评论测试
                test_comments = [
                    "This movie is absolutely amazing and brilliant!",
                    "I hated every single moment of this terrible film.",
                    "The acting was okay, but the plot was confusing.",
                    "A masterpiece of cinematography and storytelling!"
                ]

                print("\n预设评论测试:")
                for idx, comment in enumerate(test_comments, 1):
                    try:
                        results = predictor.predict_sentiment(comment)
                        print(f"\n评论 {idx}: {comment}")
                        print(f"📊 分析结果:")
                        for model_name, result in results.items():
                            print(f"{model_name}模型:")
                            print(f"  情感倾向: {result['sentiment']}")
                            print(f"  置信度: {result['confidence']:.2%}")
                    except Exception as e:
                        print(f"预测出错: {e}")

            elif choice == '3':
                print("感谢使用，再见! 👋")
                break

            else:
                print("无效的选项，请重新选择!")

    except Exception as e:
        print(f"发生错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
