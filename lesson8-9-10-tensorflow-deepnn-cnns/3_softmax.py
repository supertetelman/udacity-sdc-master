'''Quick softmax implementation'''
import numpy as np


def softmax(x):
    '''Compute softmax values for each sets of scores in x.'''
    return np.exp(x) / np.sum(np.exp(x), axis=0)


if __name__ == '__main__':
    logits = [3.0, 1.0, 0.2]
    print(softmax(logits))
