'''
For the Multi class Iris Dataset, implement multi class perceptron algorithm and compare it with softmax regression (SGD).
'''
import matplotlib.pyplot as plt
import numpy as np


class multi_class_perceptron(object):
    def __init__(self, learning_rate=0.0001, n_iteration=1000, file_path='dataset/Iris/'):
        self.lr = learning_rate
        self.n_iter = n_iteration
        self.path = file_path
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.classMap = {'0': [1, 0, 0], '1': [0, 1, 0], '2': [0, 0, 1]}
        self.classmap = [0, 1, 2]

    # 数据处理
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

        # for i in range(l_test.shape[0]):
        #     t = int(l_test[i])
        #     y_test[i][t] = 1

        return x_train, y_train, x_test, y_test

    def y_predict(self, x):
        # 返回所有预测函数的最大概率序号
        wx_dot = np.dot(self.w_, x)
        ml = np.argmax(wx_dot)
        # print(wx_dot, ml)
        return ml

    def fit(self):
        # :param x: list[np.array] 一维数组数据集
        # :param y: 被训练的数据集的实际结果
        x = self.x_train
        y = self.y_train
        # :w：权重
        self.w_ = np.random.randn(3, x.shape[1])
        # self.w_ = np.zeros((3, x.shape[1]))
        # :loss_：用于收集每一轮的loss
        self.loss_ = []

        for i in range(self.n_iter):
            loss = 0
            # y注意投影到3个标签
            for x_element, y_element in zip(x, y):
                y_element = int(y_element)
                for j in range(3):
                    # 更新w[k]的参数
                    ml = self.y_predict(x_element)
                    update = np.where((j == ml), 1, 0)-np.where((j == y_element), 1, 0)
                    # print('batch:{}'.format(j))
                    # print('update:{}'.format(update))
                    self.w_[j, :] -= self.lr*update*x_element
                    # print(self.lr*update*x_element)
                ml = self.y_predict(x_element)
                loss += np.dot(self.w_[ml, :], x_element) - np.dot(self.w_[y_element, :], x_element)

            self.loss_.append(loss)
            # print('Epoch:{} Loss:{}'.format(i, loss))

    def predict(self, x):
        result = np.zeros(3)
        x_dot = np.dot(self.w_, x)
        ml = np.argmax(x_dot)
        result[ml] = 1
        return result

    def test_accuracy(self):
        n = 0
        x_test = self.x_test
        y_test = self.y_test
        for i in range(x_test.shape[0]):
            res = self.predict(x_test[i])
            if res[int(y_test[i])] == 1:
                n = n + 1
            print('x:{} ,predict:{} ,y:{}'.format(x_test[i], res, y_test[i]))

        return n/x_test.shape[0]


    def grap_show(self):
        l_test = self.l_test
        p_test = self.p_test
        for i in range(len(l_test)):
            if int(l_test[i]) == 0:
                plt.scatter(p_test[i, 0], p_test[i, 1], marker='o', c='black', label='0')
            elif int(l_test[i]) == 1:
                plt.scatter(p_test[i, 0], p_test[i, 1], marker='*', c='black', label='1')
                print('^')
            elif int(l_test[i]) == 2:
                plt.scatter(p_test[i, 0], p_test[i, 1], marker='^', c='black', label='2')

        plt.show()


if __name__ == '__main__':
    mcp = multi_class_perceptron()
    mcp.fit()
    print(mcp.w_)
    print(mcp.test_accuracy())
    print(mcp.loss_)