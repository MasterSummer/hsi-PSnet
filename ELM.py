import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from numpy.linalg import pinv
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from DataLoader import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

columns_to_keep = [ 59,  92,  97,  95,  49,  28, 182,  68,  44,   8, 111, 168, 305,
    3, 140, 300,  55, 288, 149,  82, 278, 307, 312, 283, 101, 158,
    0,  21, 266, 129, 226]

class ELMClassifier(nn.Module):
    def __init__(self, n_hidden=100, learning_rate=0.01, epochs=1000):
        super(ELMClassifier, self).__init__()
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.input_layer = None
        self.output_layer = None

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def get_params(self, deep=True):
        return {"n_hidden": self.n_hidden, "learning_rate": self.learning_rate, "epochs": self.epochs}

    def forward(self, x):
        h = torch.relu(self.input_layer(x))
        y_pred = self.output_layer(h)
        return y_pred

    def fit(self, X, y):
        # 动态分配 n_features 和 n_classes
        n_features = X.shape[1]
        n_classes = len(np.unique(y))

        # 初始化网络层
        self.input_layer = nn.Linear(n_features, self.n_hidden)
        self.output_layer = nn.Linear(self.n_hidden, n_classes)

        # 将模型移动到设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = self(X)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()



# Example usage:
# n_hidden = 100  # Number of hidden neurons
# n_features = X_train.shape[1]  # Number of input features
# n_classes = len(np.unique(y_train))  # Number of output classes
# model = OP_ELMClassifier(n_hidden, n_features, n_classes)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)

# 激活函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


# 使用ELM进行K折交叉验证
def elm_k_fold_validation(X, y, hiddenNodeNum, num, k):
    assert X.shape[0] >= k, "The number of samples in X must be greater than or equal to k"

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    id = 0
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # print(y_test.shape, y_test[:5])

        elm = OP_ELMClassifier(hiddenNodeNum, num, 4)
        elm.fit(X_train, y_train)
        y_pred = elm.predict(X_test)

        # print(y_pred[:10])
        # print(y_test[:10])

        score = accuracy_score(y_test, y_pred)
        scores.append(score)

        print(id, score)
        id += 1


    return np.mean(scores), scores


# Example usage:
if __name__ == "__main__":

    activationFunc = sigmoid  # 激活函数
    k = 10  # K折交叉验证的折数

    data_loader = Data_Loader()
    data_loader.load_data()
    X, y = data_loader.get_data()
    print(X.shape, y.shape)
    n_components = X.shape[1]
    hiddenNodeNum = 50
    max_value = 0
    max_id = -1


    # # SPA
    # subsets = [columns_to_keep[:i + 1] for i in range(len(columns_to_keep))]
    # #
    # for subset in subsets:
    #     X_tmp = X[:, columns_to_keep]
    #     n_components = X_tmp.shape[1]
    #
    #     average_score, scores = elm_k_fold_validation(X_tmp, y, hiddenNodeNum, n_components, k=10)
    #     print(f"Average Score: {average_score}, Scores: {scores}")
    #     if average_score > max_value:
    #         max_value = average_score
    #         max_id = id
    #
    #     # 指定要保存的文件名
    #     filename = f"store/elm_base{hiddenNodeNum}_results.txt"
    #
    #     # 打开文件进行写入
    #     with open(filename, "a+") as file:
    #         # 写入平均分数
    #         file.write(f"Average Score: {average_score}\n")
    #         # 写入每一折的分数
    #         file.write("Scores: ")
    #         for score in scores:
    #             file.write(f"{score}, ")
    #         file.write("\n")
    #
    # print(max_id, max_value)

    # # SPA
    # #
    # # subsets = [columns_to_keep[:i+1] for i in range(len(columns_to_keep))]
    # #
    # # for subset in subsets:
    #
    # X_tmp = X[:, columns_to_keep]
    # # print(X.shape)
    # # if len(subset) <= k:
    # #     continue
    # # print(len(subset), X_tmp.shape, y.shape)
    # n_components = len(columns_to_keep)
    # average_score, scores = elm_k_fold_validation(X_tmp, y, hiddenNodeNum, n_components, k)
    # print(f"Average Score: {average_score}, Scores: {scores}")
    #
    # # 指定要保存的文件名
    # filename = f"store/elm_spa{n_components}_results.txt"
    #
    # # 打开文件进行写入
    # with open(filename, "a+") as file:
    #     # 写入平均分数
    #     file.write(f"Average Score: {average_score}\n")
    #     # 写入每一折的分数
    #     file.write("Scores: ")
    #     for score in scores:
    #         file.write(f"{score}, ")
    #     file.write("\n")

    dataloader = Data_Loader()
    dataloader.load_data_pca()
    X, y = dataloader.get_data()        
    
    print(f"X_pca.shape{X.shape}")
    average_score, scores = elm_k_fold_validation(X, y, hiddenNodeNum, n_components, k=40)
    print(f"Average Score: {average_score}, Scores: {scores}")

    # 指定要保存的文件名
    filename = f"store/elm_pca{n_components}_results.txt"

    # 打开文件进行写入
    with open(filename, "a+") as file:
        # 写入平均分数
        file.write(f"Average Score: {average_score}\n")
        # 写入每一折的分数
        file.write("Scores: ")
        for score in scores:
            file.write(f"{score}, ")
        file.write("\n")