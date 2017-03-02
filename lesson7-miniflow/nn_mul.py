'''Quick sanity test of the Add class'''

from miniflow import *

x, y, z = Input(), Input(), Input()

f = Mul(x, y, z)
f2 = Mul(x, y)

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
output = forward_pass2(f, graph)

output2 = forward_pass2(f2, graph)

# should output 200
print("{} * {} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
# should output 20
print("{} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output2))

assert output == 200
assert output2 == 20

print("Multiply Node functional.")
