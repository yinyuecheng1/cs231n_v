pass
from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

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
def affine_bn_relu_forward(x,w,b,gamma,beta,bn_param):
    """
    convenience layer:affine+batchnorm+relu
    Inputs:
    - x:input to the affine layer
    -w,b: weights for the affine layer
    gamma:scale parameter for the batchnorm
    beta:shif parameter for the batchnorm
    bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean or variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    """
    a,fc_cache=affine_forward(x,w,b)
    bn,bn_cache=batchnorm_forward(a,gamma,beta,bn_param)
    out,relu_cache=relu_forward(bn)
    cache=(fc_cache,bn_cache,relu_cache)
    return out,cache
def affine_bn_relu_backward(dout,cache):
    """
    backward pass for the convenience layer
    """
    fc_cache,bn_cache,relu_cache=cache
    dbn=relu_backward(dout,relu_cache)
    da,dgamma,dbeta=batchnorm_backward(dbn,bn_cache)
    dx,dw,db=affine_backward(da,fc_cache)
    return dx,dw,db,dgamma,dbeta
def affine_bn_relu_drop_forward(x,w,b,gamma,beta,bn_param,dropout_param):
    """
    convenient layer for affine-batchnorm-relu-dropout.
    """
    a,fc_cache=affine_forward(x,w,b)
    bn,bn_cache=batchnorm_forward(a,gamma,beta,bn_param)
    drop,relu_cache=relu_forward(bn)
    out,drop_cache=dropout_forward(drop,dropout_param)
    cache=(fc_cache,bn_cache,relu_cache,drop_cache)
    return out,cache
def affine_bn_relu_drop_backward(dout,cache):
    """
    backward pass for the convenient layer.
    """
    fc_cache,bn_cache,relu_cache,drop_cache=cache
    ddrop=dropout_backward(dout,drop_cache)
    dbn=relu_backward(ddrop,relu_cache)
    da,dgamma,dbeta=batchnorm_backward(dbn,bn_cache)
    dx,dw,db=affine_backward(da,fc_cache)
    return dx,dw,db,dgamma,dbeta
def affine_relu_drop_forward(x,w,b,dropout_param):
    a,fc_cache=affine_forward(x,w,b)
    drop,relu_cache=relu_forward(a)
    out,drop_cache=dropout_forward(drop,dropout_param)
    cache=(fc_cache,relu_cache,drop_cache)
    return out,cache
def affine_relu_drop_backward(dout,cache):
    fc_cache,relu_cache,drop_cache=cache
    ddrop=dropout_backward(dout,drop_cache)
    da=relu_backward(ddrop,relu_cache)
    dx,dw,db=affine_backward(da,fc_cache)
    return dx,dw,db

def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
