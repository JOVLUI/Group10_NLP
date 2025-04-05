import streamlit as st
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import traceback

# 设置页面配置必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="Group 10 IMDB Sentiment Analysis System at APU",
    page_icon="🎬",
    layout="centered"
)

# 在自定义CSS中添加模型结果卡片样式
st.markdown("""
<style>
.title-container {
    display: flex;
    justify-content: center;
    align-items: center;
    white-space: nowrap;
    margin-bottom: 20px;
}
.custom-title {
    font-size: 1.8rem !important;  
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.custom-button {
    background-color: #D98636;  /* 可以修改为你想要的背景颜色 */
    color: white;  /* 字体颜色 */
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 10px 2px;
    cursor: pointer;
    transition: background-color 0.3s;
}
.custom-button:hover {
    background-color: #45a049;  /* 悬停时的背景颜色 */
}
.model-card {
    border-radius: 10px;
    padding: 15px;
    margin: 10px;
    background-color: #e6e6fa;  /* 淡紫色 */
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
    border-left: 5px solid #8a2be2;  /* 深紫色边框 */
}
.model-card:hover {
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)


# 在显示所有模型结果的部分修改为：
def display_model_results(model_results):
    st.markdown("### 🌟 All model analysis results")

    # 创建三列
    col1, col2, col3 = st.columns(3)

    # 定义模型和对应的样式
    model_styles = {
        'Logistic_Regression': {
            'column': col1,
            'icon': '📊'
        },
        'Multinomial_Naive_Bayes': {
            'column': col2,
            'icon': '🧮'
        },
        'Multi-layer_Perceptron': {
            'column': col3,
            'icon': '🌐'
        }
    }

    # 遍历模型结果
    for model_name, result in model_results.items():
        # 获取对应的列和样式
        column = model_styles[model_name]['column']
        icon = model_styles[model_name]['icon']

        # 使用HTML和自定义CSS创建卡片
        model_name_display = {
            'Logistic_Regression': 'LR Model',
            'Multinomial_Naive_Bayes': 'MNB Model',
            'Multi-layer_Perceptron': 'MLP Model',
        }

        # 使用HTML和自定义CSS创建模型名称格式化
        model_name_pointName = {
            'Logistic_Regression': 'Logistic_Regression',
            'Multinomial_Naive_Bayes': 'Multinomial.Naive.Bayes',
            'Multi-layer_Perceptron': 'Multi-layer_Perceptron',
        }

        # 在遍历模型结果时更新卡片标题
        with column:
            model_display_name = model_name_display.get(model_name, model_name)
            model_point_name = model_name_pointName.get(model_name, model_name)
            st.markdown(f"""
            <div class="model-card">
                <h5>{icon} {model_display_name}</h5>
                <p>{model_point_name}</p>
                <p><strong>Sentiment ：</strong>{result['sentiment']}</p>
                <p><strong>Confidence：</strong>{result['confidence']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

class ModelSelector:
    def __init__(self, models_dir):
        """
        初始化模型选择器
        """
        self.models = {}
        self.vectorizer_path = None

        # 查找所有调优后的模型
        for filename in os.listdir(models_dir):
            if 'tuned_model.joblib' in filename:
                model_path = os.path.join(models_dir, filename)

                # 根据文件名确定模型名称
                if 'Logistic_Regression' in filename:
                    model_name = 'Logistic_Regression'
                elif 'Multi-layer_Perceptron' in filename:
                    model_name = 'Multi-layer_Perceptron'
                elif 'Multinomial_Naive_Bayes' in filename:
                    model_name = 'Multinomial_Naive_Bayes'
                else:
                    continue

                self.models[model_name] = model_path

            if 'tfidf_vectorizer.joblib' in filename:
                self.vectorizer_path = os.path.join(models_dir, filename)

        if not self.vectorizer_path:
            raise ValueError("未找到向量化器文件")


class SentimentPredictor:
    def __init__(self, model_path, vectorizer_path):
        """
        初始化情感预测器
        """
        # 下载必要的NLTK资源
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        # 加载模型和向量化器
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
        except Exception as e:
            st.error(f"Model loading error: {e}")
            raise

        # 停用词
        self.stop_words = set(stopwords.words('english'))

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
        预测情感
        """
        try:
            # 文本预处理
            processed_text = self.preprocess_text(text)

            # 文本向量化
            text_vectorized = self.vectorizer.transform([processed_text])

            # 安全地处理向量化结果
            if text_vectorized.shape[1] == 0:
                raise ValueError("文本向量化后特征为空")

            # 预测类别
            prediction = self.model.predict(text_vectorized)

            # 预测概率
            proba = self.model.predict_proba(text_vectorized)

            # 安全地获取类别和置信度
            pred_class = int(prediction[0])  # 确保转换为整数

            # 处理概率获取
            confidence = proba[0][pred_class]

            # 转换为可读结果
            sentiment = "positive" if pred_class == 1 else "negative"

            return sentiment, confidence

        except Exception as e:
            st.error(f"预测过程详细错误: {e}")
            raise


def main():
    # 使用自定义容器和类来显示标题
    st.markdown('<div class="title-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="custom-title">🎬 Group 10 IMDB Sentiment Analysis System at APU</h1>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 初始化模型选择器
    try:
        model_selector = ModelSelector('models')
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return

    # 文本输入
    user_input = st.text_area(
        "Enter your film comments here",
        height=200,
        placeholder="Please enter your comments about the film..."
    )

    # 预测按钮
    if st.markdown('<button class="custom-button">Analysing Emotion</button>', unsafe_allow_html=True):
        if user_input.strip():
            try:
                # 存储每个模型的结果
                model_results = {}

                # 遍历所有模型
                for model_name, model_path in model_selector.models.items():
                    predictor = SentimentPredictor(
                        model_path=model_path,
                        vectorizer_path=model_selector.vectorizer_path
                    )
                    sentiment, confidence = predictor.predict_sentiment(user_input)
                    model_results[model_name] = {
                        'sentiment': sentiment,
                        'confidence': confidence
                    }

                # 选择置信率最高的模型
                best_model = max(model_results.items(), key=lambda x: x[1]['confidence'])

                # 显示最佳模型结果
                st.markdown("### 📊 Best model analysis results")
                if best_model[1]['sentiment'] == "positive":
                    st.success(f"Model：{best_model[0]}")
                    st.success(f"Sentiment analysis result：{best_model[1]['sentiment']} 👍")
                else:
                    st.warning(f"Model：{best_model[0]}")
                    st.warning(f"Sentiment analysis result：{best_model[1]['sentiment']} 😡")

                # 显示置信度
                st.info(f"Confidence level ：{best_model[1]['confidence']:.2%}")

                # 额外信息
                st.markdown("💡 Note: The analysis is based on machine learning models and is for reference only")

                # 显示所有模型结果
                display_model_results(model_results)

            except Exception as e:
                st.error(f"分析出错: {e}")
        else:
            st.warning("请输入评论内容")

if __name__ == "__main__":
    main()
