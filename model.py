import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import rand as sprand

# define models
class MatrixFactorization:
  def __init__(self, R, k, lr=.0003, l2=.04, seed=777):
    self.R = tf.convert_to_tensor(R, dtype=tf.float32)
    self.mask = tf.not_equal(self.R, 0)
    self.m, self.n = R.shape
    self.k = k
    self.lr = lr
    self.l2 = l2
    self.tol = .001
    # Initialize trainable weights.
    self.weight_init = tf.random_normal_initializer(seed=seed)
    self.P = tf.Variable(self.weight_init((self.m, self.k)))
    self.Q = tf.Variable(self.weight_init((self.n, self.k)))

  def loss(self):
    raise NotImplementedError

  def grad_update(self):
    with tf.GradientTape() as t:
      t.watch([self.P, self.Q])
      self.current_loss = self.loss()
    gP, gQ = t.gradient(self.current_loss, [self.P, self.Q])
    self.P.assign_sub(self.lr * gP)
    self.Q.assign_sub(self.lr * gQ)

  def train(self, n_epoch=5000):
    for epoch in range(n_epoch):
      self.grad_update()
      if self.current_loss < self.tol:
        break


class BinaryMF(MatrixFactorization):
  def train(self, n_epoch=5000):
    # Cast 1/-1 as binary encoding of 0/1.
    self.labels = tf.cast(tf.not_equal(tf.boolean_mask(self.R, self.mask), -1), dtype=tf.float32)
    for epoch in range(n_epoch):
      self.grad_update()

  def loss(self):
    """Cross entropy loss."""
    logits = tf.boolean_mask(tf.matmul(self.P, self.Q, transpose_b=True), self.mask)
    logloss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=logits)
    mlogloss = tf.reduce_mean(logloss)
    l2_norm = tf.reduce_sum(self.P**2) + tf.reduce_sum(self.Q**2)
    return mlogloss + self.l2 * l2_norm


# save and load model
def save_model (model):
    model_params = {}
    
    model_params["R"] = model.R
    model_params["mask"] = model.mask
    model_params["m"] = model.m
    model_params["n"] = model.n
    model_params["lr"] = model.lr    
    model_params["l2"] = model.l2    
    model_params["tol"] = model.tol    
    model_params["weight_init"] = model.weight_init
    model_params["P"] = model.P
    model_params["Q"] = model.Q
    
    return model_params
    
def load_model (model, model_params):
    model.R = model_params["R"]
    model.mask = model_params["mask"]
    model.m = model_params["m"]
    model.n = model_params["n"]
    model.lr = model_params["lr"]
    model.l2 = model_params["l2"]
    model.tol = model_params["tol"]
    model.weight_init = model_params["weight_init"]
    model.P = model_params["P"]
    model.Q = model_params["Q"]
    
    return model

# load data
b_ratings = np.load("./small_data/200x200.npy")

# split data trian and test
rows, cols = b_ratings.nonzero()
number_of_ratings = len(b_ratings.nonzero()[0])
p = np.random.permutation(number_of_ratings)

rows, cols = rows[p], cols[p]

training_rows = rows[int(number_of_ratings/5):]
training_cols = cols[int(number_of_ratings/5):]

testing_rows = rows[:int(number_of_ratings/5)]
testing_cols = cols[:int(number_of_ratings/5)]

train_ratings = b_ratings.copy()
for i in range(len(testing_rows)):
    train_ratings[testing_rows[i], testing_cols[i]] = 0

test_ratings = b_ratings.copy()
for i in range(len(training_rows)):
    test_ratings[training_rows[i], training_cols[i]] = 0

# train model
bmf_model = BinaryMF(train_ratings, k=20, lr=0.01, l2=.0001)
bmf_model.train()

# test model
b_predictions = tf.sigmoid(tf.matmul(bmf_model.P, bmf_model.Q, transpose_b=True)).numpy()

b_mask = np.zeros_like(test_ratings)
b_mask[test_ratings.nonzero()] = 1

predictions = np.round(b_predictions * b_mask, 10)

# calculate accuracy 
print(np.average(np.rint(predictions[predictions.nonzero()]) * 2 -1 == test_ratings[predictions.nonzero()]))

# histogram of predictions (from 0 to 1) if 0 then it is very sure it is toxic and 1 it is very sure it is non-toxic and 0.5 if it has no opinion
print(np.histogram(predictions[predictions.nonzero()]))