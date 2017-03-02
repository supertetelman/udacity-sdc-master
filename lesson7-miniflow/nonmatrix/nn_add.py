'''Quick sanity test of the Add class'''

from miniflow import *

x, y, z = Input(), Input(), Input()

f = Add(x, y, z)
f2 = Add(x, y)

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

output2 = forward_pass(f2, graph)

# should output 19
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
# should output 9
print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output2))

assert output == 19
assert output2 == 9