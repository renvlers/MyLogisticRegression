import pandas as pd
import numpy as np

# 定义模型


class LogisticRegression:

    def __init__(self, learning_rate=10, num_iterations=10000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # 初始化权重和偏置
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降
        for i in range(self.num_iterations):

            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            self.learning_rate /= 1.01

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_class


class MultiClassLogisticRegression:

    def __init__(self, num_classes, num_iterations=100, tol=1e-4):
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.tol = tol
        self.classifiers = []

    def fit(self, X, y):
        # 训练每个二分类模型
        for i in range(self.num_classes):
            # 将类别i的标记设置为正类（1），其他类别的标记设置为负类（0）
            binary_y = np.where(y == i, 1, 0)
            clf = LogisticRegression()
            clf.fit(X, binary_y)
            self.classifiers.append(clf)

    def predict(self, X):
        # 对于每个样本，计算它在每个二分类模型下的预测概率
        probabilities = np.zeros((X.shape[0], self.num_classes))
        for i in range(self.num_classes):
            clf = self.classifiers[i]
            probabilities[:, i] = clf.predict(X)

        # 选择概率最大的类别作为最终预测结果
        y_pred = np.argmax(probabilities, axis=1)
        return y_pred


if __name__ == '__main__':
    # 读取西瓜数据集
    dataSet = pd.read_csv('dataSet.csv')
    dataSet = dataSet.drop('编号', axis=1)

    # 分割西瓜数据集
    train_df = dataSet.groupby('好瓜', group_keys=False).apply(
        lambda x: x.sample(frac=0.6))
    test_df = dataSet[~dataSet.index.isin(train_df.index)]
    x = dataSet.drop('好瓜', axis=1).values
    y = dataSet['好瓜'].values
    x_train = train_df.drop('好瓜', axis=1).values
    y_train = train_df['好瓜'].values
    x_test = test_df.drop('好瓜', axis=1).values
    y_test = test_df['好瓜'].values

    # 调用训练模型函数
    model = LogisticRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x)

    # 评估模型训练结果
    y_pred = model.predict(x_test)
    accuracy = sum(np.where(y_pred == y_test, 1, 0))/y_test.shape[0]
    print('模型在西瓜数据集上的精度：', accuracy, sep='')

    # 读取鸢尾花数据集
    dataSet2 = pd.read_csv('Iris.csv')
    dataSet2 = dataSet2.drop('Id', axis=1)

    # 分割鸢尾花数据集
    train_df2 = dataSet2.groupby('Species', group_keys=False).apply(
        lambda x: x.sample(frac=0.6))
    test_df2 = dataSet2[~dataSet2.index.isin(train_df2.index)]
    x2 = dataSet2.drop('Species', axis=1).values
    y2 = dataSet2['Species'].values
    x_train2 = train_df2.drop('Species', axis=1).values
    y_train2 = train_df2['Species'].values
    x_test2 = test_df2.drop('Species', axis=1).values
    y_test2 = test_df2['Species'].values

    # 预处理鸢尾花数据集
    for i in range(y2.shape[0]):
        if y2[i] == 'Iris-setosa':
            y2[i] = 0
        elif y2[i] == 'Iris-versicolor':
            y2[i] = 1
        else:
            y2[i] = 2
    for i in range(y_train2.shape[0]):
        if y_train2[i] == 'Iris-setosa':
            y_train2[i] = 0
        elif y_train2[i] == 'Iris-versicolor':
            y_train2[i] = 1
        else:
            y_train2[i] = 2
    for i in range(y_test2.shape[0]):
        if y_test2[i] == 'Iris-setosa':
            y_test2[i] = 0
        elif y_test2[i] == 'Iris-versicolor':
            y_test2[i] = 1
        else:
            y_test2[i] = 2

    # 调用训练模型函数
    model2 = MultiClassLogisticRegression(3)
    model2.fit(x_train2, y_train2)

    # 评估模型训练结果
    y_pred2 = model2.predict(x_test2)
    accuracy2 = sum(np.where(y_pred2 == y_test2, 1, 0))/y_test2.shape[0]
    print('模型在鸢尾花数据集上的精度：', accuracy2, sep='')
