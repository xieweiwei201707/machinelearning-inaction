import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.datasets.samples_generator import make_blobs

from sklearn.metrics import accuracy_score

class DecisionStump:

    def __init__(self):
        self.label = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

class Adaboost:
    # 弱分类器个数
    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
    # AdaBoost拟合算法
    def fit(self, X, y):
        m, n = X.shape
        # (1)初始化权重分布为均匀分布1/N
        w = np.full(m, (1/m))
        # 初始化基分类器列表
        self.estimators = []
        # (2) for m in (1,2,...,M)
        for _ in range(self.n_estimators):
            # (2.a) 训练一个弱分类器：决策树桩
            estimator = DecisionStump()
            # 设定一个最小化误差率
            min_error = float('inf')
            # 遍历数据集特征，根据最小分类误差率选择最优特征
            for i in range(n):
                # 获取特征值
                values = np.expand_dims(X[:, i], axis=1)
                # 特征取值去重
                unique_values = np.unique(values)
                # 尝试将每一个特征值作为分类阈值
                for threshold in unique_values:
                    p = 1
                    # 初始化所有预测值为1
                    pred = np.ones(np.shape(y))
                    # 小于分类阈值的预测值为-1
                    pred[X[:, i] < threshold] = -1
                    # (2.b) 计算误差率
                    error = sum(w[y != pred])
                    # 如果分类误差率大于0.5，则进行正负预测翻转
                    # 例如 error = 0.6 => (1 - error) = 0.4
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    # 一旦获得最小误差率，则保存相关参数配置
                    if error < min_error:
                        estimator.label = p
                        estimator.threshold = threshold
                        estimator.feature_index = i
                        min_error = error
            # (2.c) 计算基分类器的权重
            estimator.alpha = 0.5 * np.log((1.0 - min_error) /
                                           (min_error + 1e-9))
            # 初始化所有预测值为1
            preds = np.ones(np.shape(y))
            # 获取所有小于阈值的负类索引
            negative_idx = (estimator.label * X[:, estimator.feature_index] < estimator.label *
                            estimator.threshold)
            # 将负类设为'-1'
            preds[negative_idx] = -1
            # (2.d) 更新样本权重
            w *= np.exp(-estimator.alpha * y * preds)
            w /= np.sum(w)
            # 保存该弱分类器
            self.estimators.append(estimator)
    # 定义预测函数
    def predict(self, X):
        m = len(X)
        y_pred = np.zeros((m, 1))
        # 计算每个弱分类器的预测值
        for estimator in self.estimators:
            # 初始化所有预测值为1
            predictions = np.ones(np.shape(y_pred))
            # 获取所有小于阈值的负类索引
            negative_idx = (estimator.label * X[:, estimator.feature_index] < estimator.label *
                            estimator.threshold)
            # 将负类设为'-1'
            predictions[negative_idx] = -1
            # (2.e) 对每个弱分类器的预测结果进行加权
            y_pred += estimator.alpha * predictions
        # 返回最终预测结果
        y_pred = np.sign(y_pred).flatten()
        return y_pred


if __name__ == "__main__":
    X, y = make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.2, random_state=40)
    # 将标签转换为1/-1
    y_ = y.copy()
    y_[y_ == 0] = -1
    y_ = y_.astype(float)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y_, test_size=0.3, random_state=43)
    # 创建Adaboost模型实例
    clf = Adaboost(n_estimators=5)
    # 模型拟合
    clf.fit(X_train, y_train)
    # 模型预测
    y_pred = clf.predict(X_test)
    # 计算模型预测的分类准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of AdaBoost by numpy:", accuracy)