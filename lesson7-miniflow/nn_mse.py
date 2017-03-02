'''Sanity check of MSE function'''

import numpy as np
from miniflow import *

y, a = Input(), Input()
cost = MSE(y, a)

y_ = np.array([1, 2, 3])
a_ = np.array([4.5, 5, 10])

feed_dict = {y: y_, a: a_}
graph = topological_sort(feed_dict)
# forward pass
forward_pass(graph)

print(cost.value)
expected = 23.4166666667

np.testing.assert_almost_equal(cost.value, expected) # TODO: Verify this is working

print("MSE Node functional.")
