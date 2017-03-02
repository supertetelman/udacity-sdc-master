'''
Miniflow implementation that does not use matrices or numpy.

This package should not be used and is merely a demonstration of the step by step
approach used to build this tool.
'''

from functools import reduce


class Node(object):
    '''Base Node class to be subclassed by specific node types'''
    def __init__(self, inbound_nodes = []):
        '''Initialize list of inbound nodes, update outbound nodes, default value'''
        self.value = None # Default node value
        self.inbound_nodes = inbound_nodes 
        self.outbound_nodes = [] # Default output values to empty

        # Add self as an outpout for all node inputs
        for node in inbound_nodes:
            node.outbound_nodes.append(self)

    def forward(self):
        '''Forward propagation function. Must be defined in subclass'''
        raise NotImplemented


class Input(Node):
    '''Input class of node, simply passes forward the values specified'''
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        '''Take a value and set it to self.value'''
        if value is not None:
            self.value = value


class MultiInbound(Node):
    '''Base Node that allows input to be specified as (x,y,z) during init'''
    def __init__(self, *inbound_nodes):
        # Transform Set of inbound nodes to a list
        Node.__init__(self, [node for node in inbound_nodes])


class Add(MultiInbound):
    '''A class of nodes that will sum all inbound nodes'''
    def forward(self):
        '''Iterate through all inbound nodes and sum their values'''
        self.value = sum([node.value for node in self.inbound_nodes])


class Mul(MultiInbound):
    '''A class of nodes that will mutiply inbound nodes'''
    def forward(self):
        '''Iterate through all inbound nodes and multiply their values'''
        self.value = reduce(lambda x,y: x*y, [node.value for node in self.inbound_nodes])


class Lin(Node):
    '''A class of nodes that stores weights and biases and performs a linear transform'''
    def __init__(self, inputs, weights, bias):
        '''Initialize from list of inputs, list of weights, and bias value
        inbound_nodes[0] == inputs list, inbound_nodes[1] == weights list, inbound_nodes == bias
        '''
        Node.__init__(self, [ inputs, weights, bias])

    def forward(self):
        lin_sum = self.inbound_nodes[2].value
        for x, w in zip(self.inbound_nodes[0].value, self.inbound_nodes[1].value):
            lin_sum += x * w
        self.value = lin_sum


def topological_sort(feed_dict):
    '''
    Sort the nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Node and the value is the respective value feed to that Node.

    Returns a list of sorted nodes.
    '''
    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    '''
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    '''
    for n in sorted_nodes:
        n.forward()
    return output_node.value
