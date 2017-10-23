# 线性回归案例，找到一条直线，使得每一个数据点到该直线的距离最小
import numpy as np
import pylab
# 画图显示结果
def plot_data(data, b, m):
    x = data[:, 0]
    y = data[:, 1]
    y_predict = m * x + b
    pylab.plot(x, y, 'o')
    pylab.plot(x, y_predict, 'k-')
    pylab.show()
# 线性回归模型
def Linear_regression():
    data = np.loadtxt('./data/data.csv', delimiter=',')
    learning_rate = 0.001
    initial_b = 0.0
    initial_m = 0.0
    num_iter = 10000
    [b, m] = optimizer(data, initial_b, initial_m, learning_rate, num_iter)
    print('final m is {0}, b is {1}'.format(m, b))
    plot_data(data, b, m)
# 梯度下降迭代流程
def optimizer(data,starting_b, starting_m, learning_rate, num_iter):
    b = starting_b
    m = starting_m
    for i in range(num_iter):
        b, m = compute_gradient(b, m, data, learning_rate)
        if i % 1000 == 0:
            print('iter {0},error is {1}'.format(i, compute_error(b, m, data)))
    return [b, m]
# 计算梯度，更新参数
def compute_gradient(b_current, m_current, data ,learning_rate):
    N = float(len(data))
    x = data[:, 0]
    y = data[:, 1]
    b_gradient = -(2 / N) * (y - m_current * x - b_current)  # loss / b偏导数
    b_gradient = np.sum(b_gradient, axis=0)
    m_gradient = -(2 / N) * x * (y - m_current * x - b_current)  # loss / m偏导数
    m_gradient = np.sum(m_gradient, axis=0)
    # 更新参数
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]
# 计算当前误差（最小二乘法）
def compute_error(b, m, data):
    x = data[:, 0]
    y = data[:, 1]
    totalError = (y - m * x - b) ** 2
    totalError = np.sum(totalError, axis=0)
    return totalError / float(len(data))
if __name__ =='__main__':
    Linear_regression()
