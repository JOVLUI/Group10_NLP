import numpy as np
from pathlib import Path
import joblib
import time
import sys
import threading
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import uniform, randint

class ModelTuning:
    def __init__(self, data_path):
        """
        初始化模型调优器

        :param data_path: processed_data.npz 文件路径
        """
        # 加载预处理数据
        data = np.load(data_path, allow_pickle=True)

        # 将 float64 转为 float32 以减少内存占用
        self.X_train = data['X_train'].astype(np.float32)
        self.X_test = data['X_test'].astype(np.float32)
        self.y_train = data['y_train']
        self.y_test = data['y_test']

        # 设置模型保存路径
        self.model_save_path = Path(data_path).parent.parent / 'models'
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # 定义评分指标
        self.scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='binary'),
            'recall': make_scorer(recall_score, average='binary'),
            'f1': make_scorer(f1_score, average='binary')
        }

    def _save_model(self, model, model_name):
        """
        保存调优后的模型

        :param model: 训练好的模型
        :param model_name: 模型名称
        """
        save_path = self.model_save_path / f'{model_name}_tuned_model.joblib'

        try:
            joblib.dump(model, save_path)
            print(f"\n{model_name}模型已保存至 {save_path}")
        except Exception as e:
            print(f"\n保存{model_name}模型时出错: {e}")

    def _countdown_timer(self, stop_event):
        """
        倒计时计时器，用于模型调优过程中显示秒数

        :param stop_event: 停止事件
        """
        start_time = time.time()
        while not stop_event.is_set():
            elapsed_time = int(time.time() - start_time)
            sys.stdout.write(f"\r调优已进行 {elapsed_time} 秒...")
            sys.stdout.flush()
            time.sleep(1)

    def _tune_model_grid_search(self, model, param_grid, model_name):
        """
        使用网格搜索调优模型

        :param model: 待调优模型
        :param param_grid: 超参数网格
        :param model_name: 模型名称
        :return: 最佳模型
        """
        print(f"\n开始使用网格搜索调优 {model_name} 模型...")

        # 创建停止事件
        stop_event = threading.Event()

        # 启动倒计时线程
        timer_thread = threading.Thread(target=self._countdown_timer, args=(stop_event,))
        timer_thread.start()

        # 记录开始时间
        start_time = time.time()

        try:
            # 网格搜索
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring=self.scoring,
                refit='f1',
                verbose=2,
                n_jobs=-1
            )

            # 执行网格搜索
            grid_search.fit(self.X_train, self.y_train)
        finally:
            # 停止倒计时线程
            stop_event.set()
            timer_thread.join()

        # 计算调优时间
        tuning_time = time.time() - start_time

        # 清除最后一行的倒计时显示
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()

        return self._process_tuning_results(grid_search, model_name, tuning_time)

    def _tune_model_random_search(self, model, param_dist, model_name, n_iter=10):
        """
        使用随机搜索调优模型

        :param model: 待调优模型
        :param param_dist: 超参数分布
        :param model_name: 模型名称
        :param n_iter: 随机搜索的参数组合数
        :return: 最佳模型
        """
        print(f"\n开始使用随机搜索调优 {model_name} 模型...")

        # 创建停止事件
        stop_event = threading.Event()

        # 启动倒计时线程
        timer_thread = threading.Thread(target=self._countdown_timer, args=(stop_event,))
        timer_thread.start()

        # 记录开始时间
        start_time = time.time()

        try:
            # 随机搜索
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=3,
                scoring=self.scoring,
                refit='f1',
                verbose=1,
                n_jobs=-1,
                random_state=42
            )

            # 执行随机搜索
            random_search.fit(self.X_train, self.y_train)
        finally:
            # 停止倒计时线程
            stop_event.set()
            timer_thread.join()

        # 计算调优时间
        tuning_time = time.time() - start_time

        # 清除最后一行的倒计时显示
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()

        return self._process_tuning_results(random_search, model_name, tuning_time)

    def _process_tuning_results(self, search_result, model_name, tuning_time):
        """
        处理调优结果

        :param search_result: 搜索结果
        :param model_name: 模型名称
        :param tuning_time: 调优时间
        :return: 最佳模型
        """
        # 打印最佳参数和最佳得分
        print(f"\n{model_name}最佳超参数:")
        for param, value in search_result.best_params_.items():
            print(f"{param}: {value}")

        print("\n最佳验证集性能:")
        for scoring_name, scoring_func in self.scoring.items():
            score_key = f'mean_test_{scoring_name}'
            if score_key in search_result.cv_results_:
                print(f"{scoring_name}: {search_result.cv_results_[score_key][search_result.best_index_]:.4f}")

        # 使用最佳模型在测试集上评估
        best_model = search_result.best_estimator_
        test_pred = best_model.predict(self.X_test)

        print("\n测试集性能:")
        print(f"准确率: {accuracy_score(self.y_test, test_pred):.4f}")
        print(f"精确率: {precision_score(self.y_test, test_pred, average='binary'):.4f}")
        print(f"召回率: {recall_score(self.y_test, test_pred, average='binary'):.4f}")
        print(f"F1分数: {f1_score(self.y_test, test_pred, average='binary'):.4f}")

        print(f"\n{model_name}调优总时间: {tuning_time:.2f} 秒")

        # 保存最佳模型
        self._save_model(best_model, model_name.replace(' ', '_'))

        return best_model

    def tune_logistic_regression(self):
        """
        调优逻辑回归模型的超参数
        """
        # 网格搜索参数网格
        grid_param_grid = {
            'penalty': ['l2'],
            'C': [0.001, 0.01, 0.1, 1, 10],
            'solver': ['liblinear'],
            'max_iter': [100, 500, 1000]
        }

        # 随机搜索参数分布
        random_param_dist = {
            'penalty': ['l2'],
            'C': uniform(0.001, 10),
            'solver': ['liblinear'],
            'max_iter': randint(100, 1000)
        }

        # 初始化模型
        lr = LogisticRegression(random_state=42)

        # 选择调优方法：取消注释对应的方法
        return self._tune_model_grid_search(lr, grid_param_grid, 'Logistic Regression')
        # return self._tune_model_random_search(lr, random_param_dist, 'Logistic Regression', n_iter=20)

    def tune_multinomial_nb(self):
        """
        调优多项式朴素贝叶斯模型的超参数
        """
        # 网格搜索参数网格
        grid_param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1, 10],
            'fit_prior': [True, False]
        }

        # 随机搜索参数分布
        random_param_dist = {
            'alpha': uniform(0.001, 10),
            'fit_prior': [True, False]
        }

        # 初始化模型
        mnb = MultinomialNB()

        # 选择调优方法：取消注释对应的方法
        return self._tune_model_grid_search(mnb, grid_param_grid, 'Multinomial Naive Bayes')
        # return self._tune_model_random_search(mnb, random_param_dist, 'Multinomial Naive Bayes', n_iter=10)

    def tune_mlp(self):
        """
        调优多层感知器模型的超参数
        """
        # 网格搜索参数网格
        grid_param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant'],
        }

        # 随机搜索参数分布
        random_param_dist = {
            'hidden_layer_sizes': [(50,), (100,)],  # 减少配置
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': uniform(0.0001, 0.01),
            'learning_rate': ['constant'],
            'max_iter': [200, 500]  # 限制迭代次数
        }

        # 初始化模型
        mlp = MLPClassifier(max_iter=1000, random_state=42)

        # 选择调优方法：取消注释对应的方法
        return self._tune_model_grid_search(mlp, grid_param_grid, 'Multi-layer Perceptron')
        # return self._tune_model_random_search(mlp, random_param_dist, 'Multi-layer Perceptron', n_iter=15)

def main():
    # 获取项目根目录
    project_root = Path(__file__).parent.parent

    # 构建数据路径
    data_path = project_root / 'data' / 'processed_data.npz'

    # 初始化模型调优器
    tuner = ModelTuning(data_path)

    # 执行模型调优
    # 根据需要取消注释相应模型的调优
    # tuner.tune_logistic_regression()
    # tuner.tune_multinomial_nb()
    tuner.tune_mlp()

if __name__ == "__main__":
    main()
