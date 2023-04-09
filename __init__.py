import math
from random import random

import numpy as np
import pandas as pd

class Logistic:
    def __init__(self):
        self.csv_data = None   # 数据
        self.test_data = None  # 测试集
        self.train_data = None  # 训练集
        self.times = 1000   # 迭代次数
        self.a = 1 / self.times ** 0.8  # 步长阿尔法
        self.w = None # 结果矩阵

    def sigmoid(self, z):
        return 1 / (1 + math.e ** ((-1) * z))

    def read_data(self):
        self.csv_data = pd.read_csv("./作业数据集.csv")
        self.result = self.csv_data["Cured"]
        self.csv_data = self.csv_data[self.csv_data.columns[:-1]]
        self.train_data = int(len(self.csv_data) / 3 * 2)
        self.train_data = self.csv_data[:int(len(self.csv_data) / 3 * 2)]
        # self.test_data = len(self.csv_data) - len(self.train_data)
        self.test_data = self.csv_data[len(self.train_data):]
        # self.w = np.transpose(np.array([[1] * 12]))

    def classify(self):
        m, n = np.shape(self.train_data)
        self.w = np.ones((1, n))
        # self.sigmoid(self.w * )
        for i in range(self.times):
            # self.w = np.linalg.solve(self.csv_data.get, b)
            hx = self.sigmoid(np.sum(np.multiply(np.array(self.train_data), self.w), axis=1))
            diff = -np.array(self.result[:int(len(self.csv_data) / 3 * 2)]) + hx
            self.w = self.w - np.array([np.dot(diff, np.array(self.train_data))]) * self.a
                # print(1)

    def predict(self):
        m, n = np.shape(self.test_data)
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        accuracy = 0
        for i in range(m):
            hx = self.sigmoid(np.sum(np.multiply(self.w, np.array(self.test_data[i:i+1]))))
            if self.result[i + len(self.train_data)] == 1:
                if abs(hx - self.result[i + len(self.train_data)]) < 0.0000001:
                    TP += 1
                else:
                    FP += 1
            else:
                if abs(hx - self.result[i + len(self.train_data)]) < 0.0000001:
                    TN += 1
                else:
                    FN += 1
        print("准确率为：{:.2f}".format((TP+TN) / m))
        print("精确率为：{:.2f}".format(TP / (TP + FP)))
        print("召回率为：{:.2f}".format(TP / (TP + FN)))
        print("F1-score为：{:.2f}".format(2 * (TP / (TP + FP)) * (TP / (TP + FN)) / ((TP / (TP + FN)) + (TP / (TP + FP)))))

    def run(self):
        self.read_data()
        self.classify()
        self.predict()
