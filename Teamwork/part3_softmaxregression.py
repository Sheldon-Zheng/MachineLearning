'''
For the Multi class Iris Dataset, implement multi class perceptron algorithm and compare it with softmax regression (SGD).
'''
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2

class softmax_regression_SGD(object):
    def __init__(self, learning_rate=0.0001, n_iteration=1000, file_path='dataset/Iris/'):
        self.lr = learning_rate
        self.n_iter = n_iteration
        self.path = file_path
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.w_ = np.random.randn(3, self.x_train.shape[1])
        # self.w_ = np.zeros((3, x.shape[1]))
        self.batch_size = 10

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

        y_train = l_train
        y_test = l_test

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

        # :loss_：用于收集每一轮的loss
        self.loss_ = []

        for i in range(self.n_iter):
            loss = 0
            # SGD方法
            for k in range(self.batch_size):
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

    def y_predict(self, x):
        # 返回所有预测函数的最大概率序号
        wx_dot = np.dot(self.w_, x)
        ml = np.argmax(self.sigmoid(wx_dot))
        # print(wx_dot, ml)
        return ml


    def OnevsAll(self):
        x = self.x_test
        y = self.y_test
        n = 0
        for i in range(y.shape[0]):
            p = self.y_predict(x[i])
            print('x:{},predict:{},y:{}'.format(x[i], p, int(y[i])))
            if np.where((p == int(y[i])), True, False):
                n += 1

        print(n/y.shape[0])

    def predict_all(self, x):
        t = np.ones((x.shape[0], 1))
        x = np.c_[t, x]
        res_all = []
        for i in range(x.shape[0]):
            res_all.append(self.y_predict(x[i]))

        return res_all

    def fig2data(self, fig):
        import PIL.Image as Image
        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)

        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tobytes())
        image = np.asarray(image)
        return image

    def grap_show(self):
        x_test = self.x_test
        y_test = self.y_test

        x_min, x_max = x_test[:, 1].min() - 0.1, x_test[:, 1].max() + 0.1
        y_min, y_max = x_test[:, 2].min() - 0.1, x_test[:, 2].max() + 0.1

        xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        Z = np.array(self.predict_all(np.c_[xx1.ravel(), xx2.ravel()]))

        Z = Z.reshape(xx1.shape)
        fig = plt.figure()
        plot = fig.add_subplot(111)

        plot.pcolormesh(xx1, xx2, Z, cmap=plt.cm.RdYlBu, shading='auto')
        plot.contour(xx1, xx2, Z, cmap=plt.cm.Paired)

        for i in range(len(y_test)):
            if int(y_test[i]) == 0:
                plot.scatter(x_test[i, 1], x_test[i, 2], marker='o', c='black', label='0')
            elif int(y_test[i]) == 1:
                plot.scatter(x_test[i, 1], x_test[i, 2], marker='*', c='black', label='1')
                print('^')
            elif int(y_test[i]) == 2:
                plot.scatter(x_test[i, 1], x_test[i, 2], marker='^', c='black', label='2')

        image = self.fig2data(fig)
        plt.close()
        cv2.imshow('Softmaxregression', image)
        cv2.waitKey(0)

if __name__ == '__main__':
    srs = softmax_regression_SGD()
    srs.fit()
    print(srs.w_)
    srs.OnevsAll()
    print(srs.loss_)
    srs.grap_show()

