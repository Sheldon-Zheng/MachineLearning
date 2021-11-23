'''
For the Multi class Iris Dataset, implement multi class perceptron algorithm and compare it with softmax regression (SGD).
'''
import matplotlib.pyplot as plt
import numpy as np
import cv2


class multi_class_perceptron(object):
    def __init__(self, learning_rate=0.0001, n_iteration=1000, file_path='dataset/Iris/'):
        self.lr = learning_rate
        self.n_iter = n_iteration
        self.path = file_path
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.classMap = {'0': [1, 0, 0], '1': [0, 1, 0], '2': [0, 0, 1]}
        self.classmap = [0, 1, 2]
        self.w_ = np.random.randn(3, self.x_train.shape[1])
        # self.w_ = np.zeros((3, x.shape[1]))

    # 数据处理
    def load_data(self):
        path = self.path
        x_train = np.loadtxt(path + 'train/x.txt')
        l_train = np.loadtxt(path + 'train/y.txt')
        x_test = np.loadtxt(path + 'test/x.txt')
        y_test = np.loadtxt(path + 'test/y.txt')

        # 统一为wx的形式
        t = np.ones((x_train.shape[0], 1))
        x_train = np.c_[t, x_train]
        t = np.ones((x_test.shape[0], 1))
        x_test = np.c_[t, x_test]

        # 调整输出label形式
        y_train = l_train
        y_test = y_test

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

        # :loss_：用于收集每一轮的loss
        self.loss_ = []

        for i in range(self.n_iter):
            loss = 0
            # GD方法
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

        # n_x = np.arange(1, self.n_iter+1, 1)
        # plt.title('Loss Function')
        # plt.xlabel('Iteration Times')
        # plt.ylabel('Loss')
        # plt.plot(n_x, self.loss_)
        # plt.show()


    def predict_all(self, x):
        t = np.ones((x.shape[0], 1))
        x = np.c_[t, x]
        res_all = []
        for i in range(x.shape[0]):
            res_all.append(self.y_predict(x[i]))

        return res_all

    def test_accuracy(self):
        n = 0
        x_test = self.x_test
        y_test = self.y_test
        for i in range(x_test.shape[0]):
            res = self.y_predict(x_test[i])
            if res == int(y_test[i]):
                n += 1
            print('x:{} ,predict:{} ,y:{}'.format(x_test[i], res, y_test[i]))

        return round(100*n/x_test.shape[0], 2)

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
        cv2.imshow('Multiperceptron', image)
        cv2.waitKey(0)

if __name__ == '__main__':
    mcp = multi_class_perceptron()
    mcp.fit()
    print(mcp.w_)
    print(str(mcp.test_accuracy()) + '%')
    # print(mcp.predict_all())
    mcp.grap_show()