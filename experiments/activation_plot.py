import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def swish(x):
    return x * sigmoid(x)


x = np.arange(-3, 3, 0.1)
relu_y = relu(x)

sigmoid_y = sigmoid(x)
swish_y = swish(x)

plt.plot(x, relu_y, label='Relu')
plt.plot(x, sigmoid_y, label='Sigmoid')
plt.plot(x, swish_y, label='Swish')
plt.legend()
plt.grid(True)
plt.title('Activation function')
plt.savefig('./experiments/activation_functions_plot.png', dpi=600)
plt.show()