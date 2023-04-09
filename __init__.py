import math
from random import random

import numpy as np
import pandas as pd

class Logistic:
    def __init__(self):
        self.csv_data = None   # 数据
        self.test_data = None  # 测试集
        self.train_data = None  # 训练集
        self.times = 50   # 迭代次数
        self.a = 1 / self.times ** 0.8  # 步长阿尔法
        self.w = None # 结果矩阵
        self.target = None # 梯度下降
        self.o1 = None
        self.o2 = 0.5   # 初始化为0.5
        self.b = []
    def sigmoid(self, z):
        return 1 / (1 + math.e ** ((-1) * z))

    def linear(self, x1, x2, o1, o2):
        return o1 * x1 + o2 * x2

    def cost(self):
        pass

    def read_data(self):
        self.csv_data = pd.read_csv("./作业数据集.csv")
        self.result = self.csv_data["Cured"]
        self.csv_data = self.csv_data[self.csv_data.columns[:-1]]
        self.train_data = int(len(self.csv_data) / 3 * 2)
        self.train_data = self.csv_data[:int(len(self.csv_data) / 3 * 2)]
        self.test_data = len(self.csv_data) - self.train_data
        self.test_data = self.csv_data[:len(self.csv_data) - len(self.train_data)]
        # self.w = np.transpose(np.array([[1] * 12]))

    def classify(self):
        m, n = np.shape(self.train_data)
        self.w = np.ones((n, 1))

        # self.sigmoid(self.w * )
        for i in range(self.times):
            for j in range(m):
            # self.w = np.linalg.solve(self.csv_data.get, b)
                hx = self.sigmoid(np.sum(np.multiply(self.w, self.train_data[j:j+1].T))[0])
                diff = self.result[j] - hx
                self.w = self.w + self.a * diff * self.train_data[j:j+1].T
                # print(1)

    def stocGradAscent1(self, dataMatrix, classLabels, numIter=150):
        m, n = np.shape(dataMatrix)
        weights = np.ones(n)  # initialize to all ones
        for j in range(numIter):
            dataIndex = list(range(m))
            for i in range(m):
                alpha = 4 / (1.0 + j + i) + 0.0001  # apha decreases with iteration, does not
                randIndex = int(np.random.uniform(0, len(dataIndex)))  # go to 0 because of the constant
                h = self.sigmoid(sum(dataMatrix[randIndex:randIndex+1] * weights))
                error = classLabels[randIndex] - h
                weights = weights + alpha * error * dataMatrix[randIndex]
                del (dataIndex[randIndex])
        return weights


    def predict(self):
        m, n = np.shape(self.test_data)
        correct = 0
        for i in range(m):
            hx = self.sigmoid(np.sum(np.multiply(self.w, self.test_data[i:i+1].T)))
            if abs(hx - self.result[i]) < 0.00001:
                correct += 1
        print("正确率为：{}".format(correct / m))

    def run(self):
        self.read_data()
        self.stocGradAscent1(self.train_data, self.result)
        self.predict()
