from dl4cv.layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU
  
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
  
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def softmax_prob(x):
    """
    :param x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    :return: softmax prob matrix
    """
    exp_sk = np.exp(x - np.max(x, axis=1, keepdims=True))
    sum_exp_sj = np.sum(exp_sk, axis=1, keepdims=True)
    return exp_sk / sum_exp_sj