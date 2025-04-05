import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class ModelEvaluator:
    def __init__(self, data_path):
        """
        初始化模型评估器

        :param data_path: processed_data.npz 文件路径
        """
        # 加载预处理数据
        data = np.load(data_path, allow_pickle=True)

        self.X_test = data['X_test']
        self.y_test = data['y_test']

        # 设置模型路径
        self.model_path = Path(data_path).parent.parent / 'models'
        self.result_path = Path(data_path).parent.parent / 'results'
        self.result_path.mkdir(parents=True, exist_ok=True)

    def _load_model(self, model_name, tuned=True):
        """
        加载模型

        :param model_name: 模型名称
        :param tuned: 是否加载调优后的模型
        :return: 加载的模型
        """
        prefix = 'tuned' if tuned else 'initial'
        model_file = self.model_path / f'{model_name}_{prefix}_model.joblib'

        try:
            model = joblib.load(model_file)
            return model
        except Exception as e:
            print(f"加载模型 {model_file} 失败: {e}")
            return None

    def _evaluate_model(self, model, model_name):
        """
        评估单个模型

        :param model: 要评估的模型
        :param model_name: 模型名称
        :return: 评估指标字典
        """
        if model is None:
            return None

        # 预测
        y_pred = model.predict(self.X_test)

        # 计算指标
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, average='binary'),
            'Recall': recall_score(self.y_test, y_pred, average='binary'),
            'F1-Score': f1_score(self.y_test, y_pred, average='binary')
        }

        # 计算混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)

        return metrics, cm

    def _plot_confusion_matrix(self, cm, model_name, tuned=True):
        """
        绘制混淆矩阵

        :param cm: 混淆矩阵
        :param model_name: 模型名称
        :param tuned: 是否为调优模型
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
        model_type = 'Tuned' if tuned else 'Initial'
        plt.title(f'{model_type} {model_name} Confusion Matrix', fontsize=15)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()

        # 保存图像
        filename = f"{model_name.lower().replace(' ', '_')}_{'tuned' if tuned else 'initial'}_confusion_matrix.png"
        plt.savefig(self.result_path / filename)
        plt.close()

    def _plot_metric_comparison(self, initial_metrics, tuned_metrics, model_name):
        """
        绘制指标对比图

        :param initial_metrics: 初始模型指标
        :param tuned_metrics: 调优模型指标
        :param model_name: 模型名称
        """
        if initial_metrics is None or tuned_metrics is None:
            return

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        initial_values = [initial_metrics[m] for m in metrics]
        tuned_values = [tuned_metrics[m] for m in metrics]

        x = range(len(metrics))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar([i - width / 2 for i in x], initial_values, width, label='Initial Model')
        plt.bar([i + width / 2 for i in x], tuned_values, width, label='Tuned Model')

        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(f'{model_name} Performance Comparison')
        plt.xticks(x, metrics)
        plt.ylim(0, 1.05)
        plt.legend(loc='lower right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        # 添加数值标签
        for i, v in enumerate(initial_values):
            plt.text(i - width / 2, v + 0.02, f"{v:.4f}", ha='center')
        for i, v in enumerate(tuned_values):
            plt.text(i + width / 2, v + 0.02, f"{v:.4f}", ha='center')

        # 保存图像
        filename = f"{model_name.lower().replace(' ', '_')}_performance_comparison.png"
        plt.savefig(self.result_path / filename)
        plt.close()

    def evaluate_logistic_regression(self):
        """评估逻辑回归模型"""
        model_name = "Logistic Regression"
        print(f"\nEvaluating {model_name}...")

        # 加载初始和调优模型
        initial_model = self._load_model("Logistic_Regression", tuned=False)
        tuned_model = self._load_model("Logistic_Regression", tuned=True)

        # 评估初始模型
        initial_metrics, initial_cm = None, None
        if initial_model:
            initial_metrics, initial_cm = self._evaluate_model(initial_model, model_name)
            print("\nInitial Model Performance:")
            for metric, value in initial_metrics.items():
                print(f"{metric}: {value:.4f}")

            # 绘制初始模型混淆矩阵
            self._plot_confusion_matrix(initial_cm, model_name, tuned=False)

        # 评估调优模型
        tuned_metrics, tuned_cm = None, None
        if tuned_model:
            tuned_metrics, tuned_cm = self._evaluate_model(tuned_model, model_name)
            print("\nTuned Model Performance:")
            for metric, value in tuned_metrics.items():
                print(f"{metric}: {value:.4f}")

            # 绘制调优模型混淆矩阵
            self._plot_confusion_matrix(tuned_cm, model_name, tuned=True)

        # 绘制性能对比图
        if initial_metrics and tuned_metrics:
            self._plot_metric_comparison(initial_metrics, tuned_metrics, model_name)

            # 计算改进百分比
            improvements = {}
            for metric in initial_metrics:
                if initial_metrics[metric] != 0:  # 避免除以零
                    improvement = (tuned_metrics[metric] - initial_metrics[metric]) / initial_metrics[metric] * 100
                    improvements[metric] = improvement

            # 打印改进百分比
            print("\nPerformance Improvement (%):")
            for metric, improvement in improvements.items():
                print(f"{metric}: {improvement:.2f}%")

    def evaluate_multinomial_nb(self):
        """评估多项式朴素贝叶斯模型"""
        model_name = "Multinomial Naive Bayes"
        print(f"\nEvaluating {model_name}...")

        # 加载初始和调优模型
        initial_model = self._load_model("Multinomial_Naive_Bayes", tuned=False)
        tuned_model = self._load_model("Multinomial_Naive_Bayes", tuned=True)

        # 评估初始模型
        initial_metrics, initial_cm = None, None
        if initial_model:
            initial_metrics, initial_cm = self._evaluate_model(initial_model, model_name)
            print("\nInitial Model Performance:")
            for metric, value in initial_metrics.items():
                print(f"{metric}: {value:.4f}")

            # 绘制初始模型混淆矩阵
            self._plot_confusion_matrix(initial_cm, model_name, tuned=False)

        # 评估调优模型
        tuned_metrics, tuned_cm = None, None
        if tuned_model:
            tuned_metrics, tuned_cm = self._evaluate_model(tuned_model, model_name)
            print("\nTuned Model Performance:")
            for metric, value in tuned_metrics.items():
                print(f"{metric}: {value:.4f}")

            # 绘制调优模型混淆矩阵
            self._plot_confusion_matrix(tuned_cm, model_name, tuned=True)

        # 绘制性能对比图
        if initial_metrics and tuned_metrics:
            self._plot_metric_comparison(initial_metrics, tuned_metrics, model_name)

            # 计算改进百分比
            improvements = {}
            for metric in initial_metrics:
                if initial_metrics[metric] != 0:  # 避免除以零
                    improvement = (tuned_metrics[metric] - initial_metrics[metric]) / initial_metrics[metric] * 100
                    improvements[metric] = improvement

            # 打印改进百分比
            print("\nPerformance Improvement (%):")
            for metric, improvement in improvements.items():
                print(f"{metric}: {improvement:.2f}%")

    def evaluate_mlp(self):
        """评估多层感知器模型"""
        model_name = "Multi-layer Perceptron"
        print(f"\nEvaluating {model_name}...")

        # 加载初始和调优模型
        initial_model = self._load_model("Multi-layer_Perceptron", tuned=False)
        tuned_model = self._load_model("Multi-layer_Perceptron", tuned=True)

        # 评估初始模型
        initial_metrics, initial_cm = None, None
        if initial_model:
            initial_metrics, initial_cm = self._evaluate_model(initial_model, model_name)
            print("\nInitial Model Performance:")
            for metric, value in initial_metrics.items():
                print(f"{metric}: {value:.4f}")

            # 绘制初始模型混淆矩阵
            self._plot_confusion_matrix(initial_cm, model_name, tuned=False)

        # 评估调优模型
        tuned_metrics, tuned_cm = None, None
        if tuned_model:
            tuned_metrics, tuned_cm = self._evaluate_model(tuned_model, model_name)
            print("\nTuned Model Performance:")
            for metric, value in tuned_metrics.items():
                print(f"{metric}: {value:.4f}")

            # 绘制调优模型混淆矩阵
            self._plot_confusion_matrix(tuned_cm, model_name, tuned=True)

        # 绘制性能对比图
        if initial_metrics and tuned_metrics:
            self._plot_metric_comparison(initial_metrics, tuned_metrics, model_name)

            # 计算改进百分比
            improvements = {}
            for metric in initial_metrics:
                if initial_metrics[metric] != 0:  # 避免除以零
                    improvement = (tuned_metrics[metric] - initial_metrics[metric]) / initial_metrics[metric] * 100
                    improvements[metric] = improvement

            # 打印改进百分比
            print("\nPerformance Improvement (%):")
            for metric, improvement in improvements.items():
                print(f"{metric}: {improvement:.2f}%")


def main():
    # 获取项目根目录
    project_root = Path(__file__).parent.parent

    # 构建数据路径
    data_path = project_root / 'data' / 'processed_data.npz'

    # 初始化模型评估器
    evaluator = ModelEvaluator(data_path)

    # 执行模型评估（根据需要注释/取消注释）
    evaluator.evaluate_logistic_regression()
    evaluator.evaluate_multinomial_nb()
    evaluator.evaluate_mlp()


if __name__ == "__main__":
    main()