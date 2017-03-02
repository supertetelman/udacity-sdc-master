def gradient_descent_update(x, gradx, learning_rate):
    '''Returns the new x value for a single gradient descent update'''
    return  x - (learning_rate * gradx)
