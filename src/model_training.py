import numpy as np
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import threading
import time


class ModelTrainer:
    def __init__(self, data_path):
        """
        初始化模型训练器

        :param data_path: processed_data.npz 文件路径
        """
        # 加载预处理数据
        data = np.load(data_path, allow_pickle=True)

        self.X_train = data['X_train']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_test = data['y_test']

        # 项目根目录
        self.project_root = Path(data_path).parent.parent

        # 初始模型保存路径
        self.initial_models_path = self.project_root / 'models'
        self.initial_models_path.mkdir(parents=True, exist_ok=True)

        # 初始化模型
        self.logistic_regression = LogisticRegression(max_iter=1000)
        self.multinomial_nb = MultinomialNB()
        self.mlp = MLPClassifier(max_iter=1000, verbose=True)

    def _save_initial_model(self, model, model_name):
        """
        保存初始模型

        :param model: 训练好的模型
        :param model_name: 模型名称
        """
        save_path = self.initial_models_path / f'{model_name}_initial_model.joblib'

        try:
            joblib.dump(model, save_path)
            print(f"\n{model_name}初始模型已保存至 {save_path}")
        except Exception as e:
            print(f"\n保存{model_name}初始模型时出错: {e}")

    def _comprehensive_evaluation(self, y_true, y_pred):
        """
        全面的模型评估

        :param y_true: 真实标签
        :param y_pred: 预测标签
        :return: 评估指标字典
        """
        # 计算各项指标
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='binary'),
            'Recall': recall_score(y_true, y_pred, average='binary'),
            'F1-Score': f1_score(y_true, y_pred, average='binary')
        }

        return metrics

    def _plot_confusion_matrix(self, cm, model_name):
        """
        绘制混淆矩阵

        :param cm: 混淆矩阵
        :param model_name: 模型名称
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.title(f'{model_name} Confusion Matrix', fontsize=15)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.close()

    def _countdown_timer(self, stop_event):
        """
        倒计时计时器，用于MLP训练过程中显示秒数

        :param stop_event: 停止事件
        """
        start_time = time.time()
        while not stop_event.is_set():
            elapsed_time = int(time.time() - start_time)
            sys.stdout.write(f"\r训练已进行 {elapsed_time} 秒...")
            sys.stdout.flush()
            time.sleep(1)

    def train_logistic_regression(self):
        """
        训练逻辑回归模型
        """
        print("\nTraining Logistic Regression model...")

        # 记录开始时间
        start_time = time.time()

        # 训练模型
        self.logistic_regression.fit(self.X_train, self.y_train)

        # 保存初始模型
        self._save_initial_model(self.logistic_regression, 'Logistic_Regression')

        # 计算训练时间
        training_time = time.time() - start_time

        # 预测
        y_pred = self.logistic_regression.predict(self.X_test)

        # 全面评估
        metrics = self._comprehensive_evaluation(self.y_test, y_pred)

        # 打印详细指标和训练时间
        print("Logistic Regression Model Evaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print(f"训练时间: {training_time:.2f} 秒")

        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)

        # 绘制混淆矩阵
        self._plot_confusion_matrix(cm, 'Logistic Regression')

        return metrics

    def train_multinomial_nb(self):
        """
        训练多项式朴素贝叶斯模型
        """
        print("\nTraining Multinomial Naive Bayes model...")

        # 记录开始时间
        start_time = time.time()

        # 训练模型
        self.multinomial_nb.fit(self.X_train, self.y_train)

        # 保存初始模型
        self._save_initial_model(self.multinomial_nb, 'Multinomial_Naive_Bayes')

        # 计算训练时间
        training_time = time.time() - start_time

        # 预测
        y_pred = self.multinomial_nb.predict(self.X_test)

        # 全面评估
        metrics = self._comprehensive_evaluation(self.y_test, y_pred)

        # 打印详细指标和训练时间
        print("Multinomial Naive Bayes Model Evaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print(f"训练时间: {training_time:.2f} 秒")

        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)

        # 绘制混淆矩阵
        self._plot_confusion_matrix(cm, 'Multinomial Naive Bayes')

        return metrics

    def train_mlp(self):
        """
        训练多层感知器模型
        """
        print("\nTraining Multi-layer Perceptron model...")

        # 创建停止事件
        stop_event = threading.Event()

        # 启动倒计时线程
        timer_thread = threading.Thread(target=self._countdown_timer, args=(stop_event,))
        timer_thread.start()

        # 记录开始时间
        start_time = time.time()

        try:
            # 训练模型
            self.mlp.fit(self.X_train, self.y_train)
        finally:
            # 停止倒计时线程
            stop_event.set()
            timer_thread.join()

        # 保存初始模型
        self._save_initial_model(self.mlp, 'Multi-layer_Perceptron')

        # 计算训练时间
        training_time = time.time() - start_time

        # 清除最后一行的倒计时显示
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()

        # 预测
        y_pred = self.mlp.predict(self.X_test)

        # 全面评估
        metrics = self._comprehensive_evaluation(self.y_test, y_pred)

        # 打印详细指标和训练时间
        print("Multi-layer Perceptron Model Evaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print(f"训练时间: {training_time:.2f} 秒")

        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)

        # 绘制混淆矩阵
        self._plot_confusion_matrix(cm, 'Multi-layer Perceptron')

        return metrics


def main():
    # 获取项目根目录
    project_root = Path(__file__).parent.parent

    # 构建数据路径
    data_path = project_root / 'data' / 'processed_data.npz'

    # 初始化模型训练器
    trainer = ModelTrainer(data_path)

    # 注释/取消注释以控制要训练的模型
    # trainer.train_logistic_regression()
    # trainer.train_multinomial_nb()
    trainer.train_mlp()


if __name__ == "__main__":
    main()
