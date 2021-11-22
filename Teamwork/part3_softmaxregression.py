'''
For the Multi class Iris Dataset, implement multi class perceptron algorithm and compare it with softmax regression (SGD).
'''
import random
import matplotlib.pyplot as plt
import numpy as np

class softmax_regression_SGD(object):
    def __init__(self, learning_rate=0.0001, n_iteration=1000, file_path='dataset/Iris/'):
        self.lr = learning_rate
        self.n_iter = n_iteration
        self.path = file_path
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()

    def load_data(self):
        path = self.path
        x_train = np.loadtxt(path + 'train/x.txt')
        l_train = np.loadtxt(path + 'train/y.txt')
        x_test = np.loadtxt(path + 'test/x.txt')
        l_test = np.loadtxt(path + 'test/y.txt')

        # 统一为wx的形式
        t = np.ones((x_train.shape[0], 1))
        x_train = np.c_[t, x_train]
        t = np.ones((x_test.shape[0], 1))
        x_test = np.c_[t, x_test]

        # 调整输出label形式
        # y_train = np.zeros((l_train.shape[0], 3))
        y_train = l_train
        y_test = l_test
        # for i in range(l_train.shape[0]):
        #     t = int(l_train[i])
        #     y_train[i][t] = 1

        # 可以暂时不处理，仅仅用于预测使用
        # for i in range(l_test.shape[0]):
        #     t = int(l_test[i])
        #     y_test[i][t] = 1

        return x_train, y_train, x_test, y_test

    def softmax(self, x, j):
        dot_x = np.dot(self.w_[j], x)
        di = 0
        for i in range(3):
            di += np.exp(np.dot(self.w_[i], x))
        h = np.exp(dot_x)/di
        return h

    def fit(self):
        x = self.x_train
        y = self.y_train
        # :w：权重
        self.w_ = np.random.randn(3, x.shape[1])
        # self.w_ = np.zeros((3, x.shape[1]))
        # :loss_：用于收集每一轮的loss
        self.loss_ = []

        for i in range(self.n_iter):
            loss = 0
            for k in range(y.shape[0]):
                n_sgd = random.randint(0, x.shape[0]-1)
                lossi = 0
                for j in range(3):
                    grad = np.where((int(y[n_sgd]) == j), 1, 0) - self.softmax(x[n_sgd], j)
                    # print('y:{}, h:{}'.format(y[n_sgd], self.softmax(x[n_sgd], j)))
                    # print('grad:{}'.format(grad))
                    self.w_[j] += self.lr*grad*x[n_sgd]
                    lossi += np.where((j == int(y[n_sgd])), 1, 0)*np.log(self.softmax(x[n_sgd], j))
            loss += lossi
            self.loss_.append(loss)

    def sigmoid(self, z):
        # 输出最终所有类别的概率
        h = np.zeros((len(z), 1))
        h = 1.0 / (1.0 + np.exp(-z))
        return h

    def OnevsAll(self):
        x = self.x_test
        y = self.y_test
        n = 0
        for i in range(y.shape[0]):
            h = np.dot(self.w_, x[i])
            p = np.argmax(h)
            print('Each class possibility:{}'.format(h))
            print('x:{},predict:{},y:{}'.format(x[i], p, int(y[i])))
            if np.where((p == int(y[i])), True, False):
                n += 1

        print(n/y.shape[0])

if __name__ == '__main__':
    srs = softmax_regression_SGD()
    srs.fit()
    print(srs.w_)
    srs.OnevsAll()
    print(srs.loss_)

