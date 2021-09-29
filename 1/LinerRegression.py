import math
import numpy as np
from matplotlib import pyplot as plt

def F(X, a, b):
    return a*X+b

def cost(x, y, a, b):
    n_xy = len(x)
    # m组数据差的平方进行求和
    sum = 0
    for i in range(n_xy):
        sum = sum + math.pow(F(x[i], a, b)-y[i], 2)

    return sum/(2*n_xy)

def gradientDescent(x, y,n_x, a, b, lr, num_iters,t_a, t_b):
    n_xy= len(n_x)
    c_history = []
    # 梯度下降运算次数
    for i in range(num_iters):
        for j in range(n_xy):
            a = a - lr*(F(n_x[j], a, b)-y[j])*n_x[j]
            b = b - lr*(F(n_x[j], a, b)-y[j])

        t_a.append(a)
        t_b.append(b-2000*a)
        t_c = cost(n_x, y, a, b)
        c_history.append(t_c)
        print("更新参数值a:%f,b:%f" % (a, b-2000*a))
        print("更新损失值:%f" % t_c)
        #plot(x, y, a, b)

    return a, b, c_history


def plot(x, y, a, b):
    n_x = np.arange(2000, 2014, 1)

    f_y = n_x*a+b-2000*a

    plt.scatter(x, y)
    plt.plot(x, f_y)
    plt.show()
    plt.pause(1)
    plt.close()

if __name__ == '__main__':

    # 基础参数设置
    learning_rate = 0.001
    num_iters = 1000

    # 输入输出
    x = []
    n_x = []
    y = []
    # y=ax+b一阶线性回归
    a = 5
    b = 5
    # 统计参数变化
    t_a = []
    t_b = []
    t_a.append(a)
    t_b.append(b)

    cost_history = []

    with open("materials/x.txt", "r") as f:
        for i in f.readlines():
            # 去掉列表中每一个元素的换行符
            i = i.strip('\n')
            # 调整默认x
            x.append(int(i))
            n_x.append(int(i)-2000)

    with open("materials/y.txt", "r") as f:
        for i in f.readlines():
            i = i.strip('\n')
            y.append(float(i))

    print(x)
    print(y)
    print("数据加载完毕...")

    r_a, r_b, cost_history = gradientDescent(x, y, n_x, a, b, learning_rate, num_iters, t_a, t_b)

    print("最终参数值a:%f,b:%f" % (r_a, r_b-2000*r_a))
    print("2014年的房价是%f" % (F(14, r_a, r_b)))

    # 损失函数展示
    plt.figure(1)
    plt.title("Cost Function")
    plt.plot(np.arange(0, num_iters, 1), cost_history)

    # 最终结果展示
    plt.figure(2)
    plt.title("Result")
    plt.scatter(x, y)
    f_y = []
    x.append(2014)
    for i in range(len(x)):
        f_y.append(r_a * x[i] + r_b - 2000 * r_a)
    plt.plot(x, f_y)
    plt.show()



