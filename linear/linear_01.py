import numpy as np
from numpy.linalg import inv
from numpy import dot
from numpy import mat

# 最小二乘法
x = mat([1, 2, 3]).reshape(3, 1)
Y = 2 * x

theta = dot(dot(inv(dot(x.T, x)), x.T), Y)
print("最小二乘法：", theta)

# 梯度下降法

theta1 = 1.
alpha = 0.1
for i in range(100):
    theta1 = theta1 + np.sum(alpha * (Y - dot(x, theta1)) * x.reshape(1, 3)) / 3.
print("梯度下降法：", theta1)
