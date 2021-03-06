'''Sanity check of Linear node using numpy arrays'''

import numpy as np
from miniflow import *

X, W, b = Input(), Input(), Input()

f = Lin(X, W, b)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

feed_dict = {X: X_, W: W_, b: b_}

graph = topological_sort(feed_dict)
output = forward_pass2(f, graph)


expected = np.array([[-9., 4.], [-9., 4.]])

print(output)
assert expected.all() == output.all() # TODO:

print("Linear Node functional.")
