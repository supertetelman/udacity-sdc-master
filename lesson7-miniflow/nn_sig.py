'''Simple network that feeds linear transform into sigmoid'''

import numpy as np
from miniflow import *

X, W, b = Input(), Input(), Input()

f = Linear(X, W, b)
g = Sigmoid(f)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

feed_dict = {X: X_, W: W_, b: b_}

graph = topological_sort(feed_dict)
output = forward_pass2(g, graph)

expected = np.array([[  .000123394576,   .982013790],
        [  .000123394576,   .982013790]])

print(output)
print(expected)
np.testing.assert_allclose(output.all(), expected.all(), rtol=1e-4, atol=4) # TODO:

print("Sigmoid Node functional.")
