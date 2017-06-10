import numpy as np


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param) for the backward pass
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    pad = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, C, HF, WF = w.shape
    H_out = 1 + (H + 2 * pad - HF) / stride
    W_out = 1 + (W + 2 * pad - WF) / stride
    X_col = im2col(x, HF, WF, stride, pad)
    W_row = w.reshape(F, -1)
    b = b.reshape(-1, 1)
    out = W_row.dot(X_col) + b
    out = out.reshape(F, N, H_out, W_out)
    out = out.transpose(1, 0, 2, 3)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    F, C, HF, WF = w.shape
    # since axis=1 corresponds to the number of filters and our biases go into
    # every filter, so we sum along the rest of the axis.
    db = np.sum(dout, axis = (0, 2, 3))

    X_col = im2col(x, HF, WF, stride, pad)
    dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
    dw = dout_reshaped.dot(X_col.T)
    dw = dw.reshape(w.shape)

    w_reshape = w.reshape(F, -1)
    dx_col = w_reshape.T.dot(dout_reshaped)
    dx = col2im(dx_col, x, HF, WF, stride, pad)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param) for the backward pass
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    N, C, H, W = x.shape
    stride = pool_param['stride']
    HP = pool_param['pool_height']
    WP = pool_param['pool_width']
    H_out = (H - HP)/stride + 1
    W_out = (W - WP)/stride + 1
    C_out = C
    x_reshaped = x.reshape(N*C, 1, H, W)
    x_col = im2col(x_reshaped, HP, WP, stride, pad = 0)
    maxIdx = np.argmax(x_col, axis = 0)
    out = x_col[maxIdx, range(maxIdx.size)]
    out = out.reshape(N, C_out, H_out, W_out)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, maxIdx, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    x, maxIdx, pool_param = cache
    N, C, H, W = x.shape
    stride = pool_param['stride']
    HP = pool_param['pool_height']
    WP = pool_param['pool_width']
    x_reshaped = x.reshape(N * C, 1, H, W)
    x_col = im2col(x_reshaped, HP, WP, stride, pad = 0)
    dx_col = np.zeros_like(x_col)
    dout_flat = dout.flatten()
    dx_col[maxIdx, range(maxIdx.size)] = dout_flat
    dx = col2im(dx_col, x_reshaped, HP, WP, stride, pad=0)
    dx = dx.reshape(x.shape)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.

  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    sample_mean = np.mean(x, axis=0)

    x_minus_mean = x - sample_mean

    sq = x_minus_mean**2

    var = 1./N * np.sum(sq, axis=0)

    sqrtvar = np.sqrt(var + eps)

    ivar = 1./sqrtvar

    x_norm = x_minus_mean * ivar

    gammax = gamma * x_norm

    out = gammax + beta

    running_var  = momentum * running_var + (1 - momentum) * var
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean

    cache = (out, x_norm, beta, gamma, x_minus_mean, ivar, sqrtvar, var, eps)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    x = (x - running_mean) / np.sqrt(running_var)
    out = x * gamma + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  N, D = dout.shape
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def im2col(x, HF, WF, stride=1, pad=1):
    padded_x = np.pad(x, pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                      mode = 'constant', constant_values = 0)
    N, C, H, W = padded_x.shape
    X_col = []
    for n in range(N):
        for h in range(0, H - HF + 1, stride):
            for wi in range(0, W - WF + 1, stride):
                patch = padded_x[n, :, h:h + HF, wi:wi + WF]
                patch_flat = patch.flatten()
                X_col.append(patch_flat)
    X_col = np.array(X_col)
    return X_col.T

def col2im(X_col, x, HF, WF, stride=1, pad=1):
    N, C, H, W = x.shape
    H_padded = H + 2*pad
    W_padded = W + 2*pad
    x_padded = np.zeros((N, C, H_padded, W_padded))
    c_index = 0
    for n in range(N):
        for h in range(0, H_padded-HF+1, stride):
            for wi in range(0, W_padded-WF+1, stride):
                patch = X_col[:, c_index]
                patch_reshaped = patch.reshape(C, HF, WF)
                x_padded[n, :, h:h + HF, wi:wi + WF] += patch_reshaped
                c_index += 1
    if pad == 0:
        return x_padded
    return x_padded[:, :, pad:-pad, pad:-pad]