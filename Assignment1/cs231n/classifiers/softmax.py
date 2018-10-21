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
  num_train = X.shape[0]
  num_classes = W.shape[1]

  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  for i in range(num_train):
        scores = X[i, :].dot(W)
        scores -= np.max(scores)
        P = np.exp(scores) / np.sum(np.exp(scores))
        loss -= np.log(P[y[i]])
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += X[i, :] * (P[j] - 1)
            else:
                dW[:, j] += X[i, :] * P[j]

  loss /= num_train
  dW /= num_train
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  num_train = X.shape[0]
  
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  scores = np.dot(X, W) 
#   scores -= np.max(scores)
  P = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

  # compute the loss: average cross-entropy loss and regularization
  correct_logprobs = -np.log(P[range(num_train),y])
  data_loss = np.sum(correct_logprobs) / num_train
  reg_loss = 0.5 * reg * np.sum(W * W)
  loss = data_loss + reg_loss
    
  # compute gradient  
  dscores = P
  dscores[range(num_train), y] -= 1
  dscores /= num_train
  dW = np.dot(np.transpose(X), dscores)
  dW += reg * W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW

