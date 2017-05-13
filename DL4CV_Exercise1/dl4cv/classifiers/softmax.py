import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
  
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
  
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
  
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass

    def calculate_softmax(cls,exp_score_vector, softmax_denominator):
        return exp_score_vector[cls]/softmax_denominator;

    for i, data_row in enumerate(X):
        score_or_prediction_vector = data_row.dot(W)
        normalized_score = score_or_prediction_vector - np.max(score_or_prediction_vector)
        exp_score_vector = np.exp(normalized_score)
        softmax_denominator = np.sum(exp_score_vector)
        actual_sample_class = y[i]
        softmax = calculate_softmax(actual_sample_class,exp_score_vector,softmax_denominator)
        loss += -np.log(softmax)

        # gradient
        for j in range(np.shape(W)[1]):
            softmax = calculate_softmax(j,exp_score_vector,softmax_denominator)
            tnj = 1 if j == y[i] else 0
            dwj = (softmax - tnj)*data_row
            dW[:, j] += dwj

    num_of_data = np.shape(X)[0]
    loss = loss / num_of_data
    loss += 0.5*reg*np.sum(W*W)
    dW  = dW / num_of_data
    dW = dW + reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
  
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass
    score_or_prediction_matrix = X.dot(W)
    normalized_matrix = score_or_prediction_matrix - np.max(score_or_prediction_matrix, axis=1, keepdims=True)

    # for overflow handling of exp
    max_score = np.max(normalized_matrix, axis=1)
    normalized_matrix = (normalized_matrix.transpose() - max_score).transpose()

    exp_score_matrix = np.exp(normalized_matrix)
    softmax_denominator_vector = np.sum(exp_score_matrix, axis=1, keepdims=True)
    softmax_matrix = exp_score_matrix / softmax_denominator_vector
    num_of_data = np.shape(X)[0]
    softmax_vector = softmax_matrix[np.arange(num_of_data), y]
    loss = np.sum(-np.log(softmax_vector))

    # gradient
    T = np.zeros_like(softmax_matrix) #T is the actual lable matrix
    T[np.arange(num_of_data), y] = 1
    dW = X.T.dot(softmax_matrix-T) # gradient with respect to weights

    loss = loss / num_of_data
    loss += 0.5 * reg * np.sum(W * W)
    dW = dW / num_of_data
    dW = dW + reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

