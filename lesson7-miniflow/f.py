''' Given any random starting point X gradient descent should be able calculate minimum cost for f(x)'''

import random
from gd import gradient_descent_update


def f(x):
    '''Simple Quadratic function'''
    return x**2 + 5


def df(x):
    '''Derivative of `f` with respect to `x`.'''
    return 2*x


# Randomly initialize X
x = random.randint(0, 10000)

learning_rate = .3 # INFO: Play with this.
epochs = 100

for i in range(epochs+1):
    cost = f(x)
    gradx = df(x)
    print("EPOCH {}: Cost = {:.3f}, x = {:.3f}".format(i, cost, gradx))
    x = gradient_descent_update(x, gradx, learning_rate)
