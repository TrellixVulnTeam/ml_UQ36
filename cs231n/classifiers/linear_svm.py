import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
  
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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]  # the score of the correct class
        for j in xrange(num_classes):
            if j == y[i]:
                pass  # skip the correct class, since sj-syi+1 will always result in loss+=1
            else:
                margin = scores[j] - correct_class_score + 1  # note delta = 1
                # Only compute the loss/gradient if the score is past a delta margin
                if margin > 0:
                    loss += margin  # otherwise, the loss is 0
                    # Compute the gradient with respect to weights
                    dW[:, j] += X[i]  # partial derivative is with respect to j
                    dW[:, y[i]] -= X[i]  # partial derivative is with respect to yi
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W  # derivative is 2*W, but let the reg hyperparameter absorb the factor of 2

    #############################################################################
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
  
    Inputs and outputs are the same as svm_loss_naive.
    """
    delta = 1.0
    loss = 0.0
    dW = np.ones(W.shape)  # initialize the gradient as zero

    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # W.shape = (3073, 10)
    # X.shape = (500, 3073)
    # y.shape = (500,)
    scores_mat = X.dot(W)  # scores_mat.shape = (500, 10)
    margins_mat = scores_mat - \
                  np.tile(scores_mat[xrange(num_train), y], (num_classes, 1)).T + \
                  delta  # margins_mat.shape = (500, 10)
    margins_mat = np.maximum(0, margins_mat)
    margins_mat[xrange(num_train), y] = 0
    loss = np.sum(margins_mat) / num_train
    loss += reg * np.sum(W*W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    num_above_margin = np.sum(margins_mat>0, axis=1)
    result_mat = np.zeros_like(margins_mat)
    result_mat[margins_mat>0] = 1
    result_mat[xrange(num_train), y] = -num_above_margin
    dW = X.T.dot(result_mat) / num_train + reg*W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
