import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


def fig2data(fig):
    import PIL.Image as Image
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    return image


def sigmoid(z):  # 激活函数之前的值称之为z
    return 1.0 / (1.0 + np.exp(-z))


def dsigmoid(z):  # sigmoid的导数
    return sigmoid(z)(1 - sigmoid(z))


class MLP_np:

    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes) - 1  # 网络层数2层
        self.weights = [np.random.randn(ch2, ch1) for ch1, ch2 in zip(sizes[:-1], sizes[1:])]  # [784, 30], [30, 10]
        self.biases = [np.random.randn(ch, 1) for ch in sizes[1:]]  # z = wx + b 因为wx是[30, 1]，所以b也是[30, 1]

    def forward(self, x):
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, x) + b  # 矩阵相乘

            # [30, 1]
            x = sigmoid(z)

        return x

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # 新建和self.weights同维度的列表，存储梯度信息

        nabla_b = [np.zeros(b.shape) for b in self.biases]  # 新建和self.biases同维度的列表，存储梯度信息

        # 1.前向传播

        zs = []  # 保存每一层输出值
        activations = [x]  # 保存每一层激活值

        activation = x  # 保存输入数据

        # 单独写forward是为了test的时候用，test不需要反向传播
        for b, w in zip(self.biases, self.weights):  # zip返回ndarray对象，b[0]w[0], b[1]w[1]

            z = np.dot(w, activation) + b  # z = wx + b，激活函数前的值称为z
            activation = sigmoid(z)  # z经过激活函数输出，作为下一层的输入

            zs.append(z)  # 记录z方便以后计算梯度
            activations.append(activation)  # 记录activation方便以后计算梯度

        loss = np.power(activations[-1] - y, 2).sum()  # 计算loss，不加sum的话返回向量

        # 2.反向传播

        # 2.1 计算输出层梯度
        delta = activations[-1] * (1 - activations[-1]) * (activations[-1] - y)  # 点乘 [10, 1] with [10, 1] => [10, 1]
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[
            -2].T)  # 矩阵[10,1]@[1,30]才会得到[10,30]，而activations[-2]维度为[30,1]，所以activations[-2]需要加一个转置

        # 2.2 计算隐藏层梯度
        z = zs[-2]
        a = activations[-2]

        # [10, 30]T @ [10, 1] => [30, 10] @ [10, 1] => [30,1] 因为公式有求和在，所以是矩阵相乘
        # [30, 1] * [30, 1] => [30, 1]
        delta = np.dot(self.weights[-1].T, delta) * a * (1 - a)

        nabla_b[-2] = delta
        # [30, 1] @ [784, 1]T => [30, 784]
        nabla_w[-2] = np.dot(delta, activations[-3].T)  # 本质上是对应位置相乘，但是因为数据存储格式不一样，所以采用了矩阵相乘的方式

        return nabla_w, nabla_b, loss

    def train(self, training_data, epochs, batchsz, lr, test_data):
        if test_data:
            n_test = len(test_data)  # 记录测试数据长度 10000
        n = len(training_data)  # 记录训练数据长度 50000

        for j in range(epochs):
            random.shuffle(training_data)  # 打散训练数据

            # 切割数据，切割成一个一个batch的，每个mini_batches包含10组训练数据
            mini_batches = [training_data[k:k + batchsz] for k in range(0, n, batchsz)]

            for mini_batch in mini_batches:
                loss = self.update_mini_batch(mini_batch, lr)  # 在每个batch上更新weights和biases，返回loss

            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test), loss)
                self.show(test_data, j == epochs - 1)
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, batch, lr):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        loss = 0

        # x:{ndarray:(784, 1)}
        # y:{ndarray:(10, 1)}
        for x, y in batch:
            nabla_w_, nabla_b_, loss_ = self.backprop(x, y)  # 返回更新后的梯度、偏置和loss

            nabla_w = [accu + cur for accu, cur in zip(nabla_w, nabla_w_)]  # 相应位置进行累加
            nabla_b = [accu + cur for accu, cur in zip(nabla_b, nabla_b_)]
            loss += loss_

        nabla_w = [w / len(batch) for w in nabla_w]  # 点除，取平均值
        nabla_b = [b / len(batch) for b in nabla_b]
        loss = loss / len(batch)

        self.weights = [w - lr * nabla for w, nabla in zip(self.weights, nabla_w)]  # w = w - lr * nabla_w 更新w
        self.biases = [b - lr * nabla for b, nabla in zip(self.biases, nabla_b)]  # b = b - lr * nabla_b 更新b

        return loss

    def evaluate(self, test_data):
        result = [(np.argmax(self.forward(x)), y) for x, y in test_data]  # y不是one-hot编码，y是标量！
        correct = sum(int(pred == y) for pred, y in result)  # 返回识别对的个数
        return correct

    def show(self, test_data, is_last):
        # np.concatenate把两数组拼接
        # np.newaxis增加数组维度
        X = np.concatenate([x[np.newaxis, :, 0] for x, y in test_data])
        y = np.array([y for x, y in test_data])

        x1_min, x1_max = np.min(X[:, 0]) - 0.1, np.max(X[:, 0]) + 0.1
        x2_min, x2_max = np.min(X[:, 1]) - 0.1, np.max(X[:, 1]) + 0.1

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
        # Z = np.array([np.argmax(self.forward(x[:, np.newaxis])) for x in np.c_[xx1.ravel(), xx2.ravel()]])
        Z = np.argmax(self.forward(np.c_[xx1.ravel(), xx2.ravel()].transpose()), 0)
        print(Z)
        print(Z.shape)
        print(xx1.shape)
        Z = Z.reshape(xx1.shape)
        fig = plt.figure()
        plot = fig.add_subplot(111)
        plot.pcolormesh(xx1, xx2, Z, cmap=plt.cm.RdYlBu, shading='auto')
        plot.contour(xx1, xx2, Z, cmap=plt.cm.Paired)
        plot.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        image = fig2data(fig)
        plt.close()
        cv2.imshow('fuch you', image)
        cv2.waitKey(0 if is_last else 1)


def main():
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    root = './dataset/Iris/'
    training_data = [(x[:, np.newaxis], np.eye(3)[int(y)][:, np.newaxis])
                     for x, y in zip(np.genfromtxt(root + 'train/x.txt'),
                                     np.genfromtxt(root + 'train/y.txt'))]
    test_data = [(x[:, np.newaxis], int(y))
                 for x, y in zip(np.genfromtxt(root + 'test/x.txt'),
                                 np.genfromtxt(root + 'test/y.txt'))]
    net = MLP_np([2, 4, 3])  # 建立神经网络
    net.train(training_data, epochs=80, batchsz=12, lr=1.0, test_data=test_data)


if __name__ == '__main__':
    main()
