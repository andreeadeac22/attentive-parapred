import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def lrelu(x):
    return np.maximum(0.2*x, x)

def elu(x):
    a = []
    for i in x:
        if i >= 0:
            a.append(i)
        else:
            a.append( (np.exp(i) - 1))
    return a


x = np.arange(-2.0, 2.0, 0.1)
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)
y_lrelu = lrelu(x)
y_elu = elu(x)

plt.plot(x, y_tanh, label='Tanh', color='g', lw=1)
plt.plot(x, y_sigmoid, label='Sigmoid', color='k', lw=1)
plt.plot(x, y_relu, label='ReLU', color='r', lw=1)
plt.plot(x, y_lrelu, label='LReLU', color='b', lw=1)
plt.plot(x, y_elu, label='ELU', color='y', lw=1)
plt.ylim(-1.1, 1.1)
plt.ylabel(r'$\sigma(a)$', fontsize = 24)
plt.xlabel(r'$a$', fontsize = 24)
plt.legend(loc="lower right")
plt.show()

plt.savefig("activations.pdf")


#from matplotlib2tikz import save as tikz_save
#tikz_save('test.tex')