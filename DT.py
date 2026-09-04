# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Data_Loader import Data_Loader


data_loader = Data_Loader()
data_loader.load_data()
X, y = data_loader.get_data()
print(X.shape, y.shape)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
conf_matrices = []

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 创建并训练模型
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # 验证模型
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    accuracies.append(accuracy)

    # 计算混淆矩阵
    # cm = confusion_matrix(y_val, y_pred)
    # conf_matrices.append(cm)

    # 打印每次迭代的准确率
    print(f"Fold {fold + 1} accuracy: {accuracy}")

# 计算平均准确率
average_accuracy = np.mean(accuracies)
print(f"Average accuracy: {average_accuracy}")

filename = f"store/DT_base_results.txt"
with open(filename, "a+") as file:
    # 写入平均分数
    file.write(f"Average Score: {average_accuracy}\n")
    # 写入每一折的分数
    file.write("Scores: ")
    for accuracy in accuracies:
        file.write(f"{accuracy}, ")
    file.write("\n")

