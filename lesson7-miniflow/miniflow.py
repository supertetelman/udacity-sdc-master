from functools import reduce
import numpy as np

class Node(object):
    '''Base Node class to be subclassed by specific node types'''
    def __init__(self, inbound_nodes = []):
        '''Initialize list of inbound nodes, update outbound nodes, default value'''
        self.value = None # Default node value
        self.inbound_nodes = inbound_nodes 
        self.outbound_nodes = [] # Default output values to empty
        self.gradients = {}

        # Add self as an outpout for all node inputs
        for node in inbound_nodes:
            node.outbound_nodes.append(self)

    def forward(self):
        '''Forward propagation function. Must be defined in subclass'''
        raise NotImplemented

    def backward(self):
        '''Backward propagation function. Must be defined in subclass'''
        raise NotImplementedError


class Input(Node):
    '''Input class of node, simply passes forward the values specified'''
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        '''Take a value and set it to self.value'''
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.outbound_nodes:
            self.gradients[self] += n.gradients[self]


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
    
    def backward(self):
        pass # TODO


class Mul(MultiInbound):
    '''A class of nodes that will mutiply inbound nodes'''
    def forward(self):
        '''Iterate through all inbound nodes and multiply their values'''
        self.value = reduce(lambda x,y: x*y, [node.value for node in self.inbound_nodes])

    def backward(self):
        pass # TODO


class Lin(Node):
    '''A class of nodes that stores weights and biases and performs a linear transform'''
    def __init__(self, inputs, weights, bias):
        '''Initialize from list of inputs, list of weights, and bias value'''
        # inbound_nodes[0] == inputs list, inbound_nodes[1] == weights list, inbound_nodes == bias
        Node.__init__(self, [ inputs, weights, bias])

    def forward(self):
        '''Returns the dot product of the weights and inputs plus the bias'''
        self.value = np.dot(self.inbound_nodes[0].value, self.inbound_nodes[1].value) + self.inbound_nodes[2].value

    def backward(self):
        '''Calculates the gradient based on the output values.'''
        # Initialize a partial for each of the inbound_nodes.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this node's inputs.
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # Set the partial of the loss with respect to this node's weights.
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # Set the partial of the loss with respect to this node's bias.
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class Linear(Lin):
    '''Long-named class for Lin'''
    pass


class Sig(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        '''Takes a numpy array X and performs the sigmoid on it'''
        return 1. / (1. + np.exp(-x))

    def forward(self):
        '''Update value to the sigmoid of the inbound node value'''
        self.value = self._sigmoid(self.inbound_nodes[0].value)

    def backward(self):
        '''Run a backwards propagation and update gradient values'''
        # Initialize gradients to 0
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            # Sum the derivative with respect to the input over all the outputs.
            self.gradients[self.inbound_nodes[0]] += self.value * (1 - self.value) * grad_cost
        return



class Sigmoid(Sig):
    '''Long-named class for Sig'''
    pass


class MSE(Node):
    '''Mean Squared Error Node (cost function)
    Should bu used as the last node for a network
    '''
    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    def forward(self):
        '''Calculate the mean squared error'''

        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        self.value = np.mean((y - a) ** 2)

        # Save the computed output for backward.
        self.m = self.inbound_nodes[0].value.shape[0]
        self.diff = y-a

    def backward(self):
        '''Calculate the gradient of the cost function'''
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff


def topological_sort(feed_dict):
    '''Sort the nodes in topological order using Kahn's Algorithm.

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


def forward_pass2(output_node, sorted_nodes):
    '''Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    '''
    for n in sorted_nodes:
        n.forward()
    return output_node.value


def forward_pass(graph):
    '''Performs a forward pass through a list of sorted Nodes.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    '''
    # Forward pass
    for n in graph:
        n.forward()


def forward_and_backward(graph):
    '''Performs a forward pass and a backward pass through a list of sorted Nodes.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    '''
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    for n in graph[::-1]:
        n.backward()


def sgd_update(trainables, learning_rate=1e-2):
    '''Updates the value of each trainable with SGD.

    Arguments:

        `trainables`: A list of `Input` Nodes representing weights/biases.
        `learning_rate`: The learning rate.
    '''
    # Performs SGD
    #
    # Loop over the trainables
    for t in trainables:
        # Change the trainable's value by subtracting the learning rate
        # multiplied by the partial of the cost with respect to this
        # trainable.
        partial = t.gradients[t]
        t.value -= learning_rate * partial
