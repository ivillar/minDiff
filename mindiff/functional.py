import numpy as np


def softmax(array, axis=None):
    shiftx = array - np.max(array, axis=axis, keepdims=True)
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=axis, keepdims=True)
