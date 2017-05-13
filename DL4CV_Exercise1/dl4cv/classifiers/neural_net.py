import numpy as np


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.
  
    In other words, the network has the following architecture:
  
    input - fully connected layer - ReLU - fully connected layer - softmax
  
    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
    
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)
    
        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.
    
        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.
    
        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].
    
        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        H, C = W2.shape

        # Compute the forward pass
        scores = np.zeros(shape=(N, C), dtype=float)
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        pass
        data = self.forward_pass(X,do_dropout=True)
        X_with_bias = data['X_with_bias']       # N,D+1
        W1_with_bias = data['W1_with_bias']     # D+1,H
        A1 = data['A1']                         # N,H
        Z1 = data['Z1']                         # N,H
        Z1_with_bias = data['Z1_with_bias']     # N,H+1
        W2_with_bias = data['W2_with_bias']     # H+1,C
        scores = data['scores']                 # N,C

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss. So that your results match ours, multiply the            #
        # regularization loss by 0.5                                                #
        #############################################################################
        pass
        softmax_matrix = self.softmax(scores)       # N,C
        softmax_vector = softmax_matrix[np.arange(N), y]

        # for handling numerical errors
        #softmax_vector = softmax_vector  + 0.000001

        loss = np.sum(-np.log(softmax_vector)) / N
        loss += 0.5 * reg * (np.sum(W1_with_bias * W1_with_bias) + np.sum(W2_with_bias * W2_with_bias))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        pass
        actual_label_matrix = np.zeros_like(softmax_matrix)
        actual_label_matrix[np.arange(N), y] = 1

        # derivative of loss/Error(E) wrt W2 (weights of second layer)
        # delta error for output layer
        dEbydZ1 = softmax_matrix - actual_label_matrix      # N,C
        dW2_with_bias = Z1_with_bias.T.dot(dEbydZ1)         # H+1,C
        dW2_with_bias = dW2_with_bias / N + reg * W2_with_bias

        # derivative of loss/Error(E) wrt W1 (weights of first layer)
        gradient_ReLU = self.dReLU(A1)              # N,H
        sum = dEbydZ1.dot(W2.T)                     # N,H | we can ignore bias here
        dEbydA1 = gradient_ReLU*sum                 # N,H | delta error for hidden layer
        dW1_with_bias = X_with_bias.T.dot(dEbydA1)  # D+1,H
        dW1_with_bias = dW1_with_bias / N + reg * W1_with_bias

        grads['W1'] = dW1_with_bias[:D, :]
        grads['b1'] = dW1_with_bias[-1]
        grads['W2'] = dW2_with_bias[:H, :]
        grads['b2'] = dW2_with_bias[-1]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.
    
        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            pass
            indices = np.random.choice(np.arange(num_train), size=batch_size, replace=True)
            X_batch = X[indices]
            y_batch = y[indices]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            pass
            self.params['W1'] += -learning_rate * grads['W1']
            self.params['b1'] += -learning_rate * grads['b1']
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b2'] += -learning_rate * grads['b2']
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.
    
        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.
    
        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        pass
        data = self.forward_pass(X,do_dropout=False)
        pred_matrix = data['scores']
        y_pred = np.argmax(pred_matrix, axis=1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred

    def ReLU(self, A1):
        return np.maximum(A1, np.zeros_like(A1))

    def dReLU(self, A1):
        return (A1 > 0).astype(float)

    def softmax(self, scores):

        exp_score_matrix = np.exp(scores)

        # handling numerical issues
        # max_score = np.max(exp_score_matrix, axis=1)
        # exp_score_matrix = (exp_score_matrix.transpose() - max_score).transpose()


        softmax_denominator_vector = np.sum(exp_score_matrix, axis=1, keepdims=True)
        softmax_matrix = exp_score_matrix / softmax_denominator_vector
        return softmax_matrix

    def droput(self, N, H):
        dropout_percent = 0.25
        dp = np.random.binomial([np.ones((N, H))], 1 - dropout_percent)[0] * (1.0 / (1 - dropout_percent))
        return dp

    def forward_pass(self, X, do_dropout):
        import scipy.stats.mstats as spy

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        H, C = W2.shape
        ones = np.ones(N)
        if np.isnan(W1).any():
            W1 = np.nan_to_num(W1)
        if np.isnan(W2).any():
            W2 = np.nan_to_num(W2)
        if np.isnan(b1).any():
            b1 = np.nan_to_num(b1)
        if np.isnan(b2).any():
            b2 = np.nan_to_num(b2)
        X_with_bias = np.column_stack((X, ones))    # N,D+1
        W1_with_bias = np.row_stack((W1, b1))       # D+1,H
        A1 = X_with_bias.dot(W1_with_bias)          # N,H
        Z1 = self.ReLU(A1)                          # N,H
        if (do_dropout):
            Z1 = Z1*self.droput(N,H)

        Z1_with_bias = np.column_stack((Z1, ones))  # N,H+1
        W2_with_bias = np.row_stack((W2, b2))       # H+1,C
        scores = Z1_with_bias.dot(W2_with_bias)     # N,C

        #for overflow handling of exp
        max_score = np.max(scores, axis=1)
        scores = (scores.transpose() - max_score).transpose()

        if np.isnan(scores).any():
            scores = np.nan_to_num(scores)

        return {
            'X_with_bias': X_with_bias,
            'W1_with_bias': W1_with_bias,
            'A1': A1,
            'Z1': Z1,
            'Z1_with_bias': Z1_with_bias,
            'W2_with_bias': W2_with_bias,
            'scores': scores
        }
