import numpy as np
import pandas as pd

X = np.load('data/train.npy')
y_ = np.load('data/label.npy')
X_test = np.load('data/test.npy')
W = np.zeros(163)

np.random.seed(0)

def split_validation(x, y, split):
   indices = np.arange(len(x))
   np.random.shuffle(indices)
   x_org = x[indices]
   y_org = y[indices]
   x_val = x_org[:int(len(x)*split)]
   y_val = y_org[:int(len(y)*split)]
   x = x_org[int(len(X)*split):]
   y = y_org[int(len(y)*split):]
   return x, x_val, y, y_val

def calculate_loss(x, y_, w):
   y = np.dot(x, w)
   loss = (np.linalg.norm(y_ - y)**2) / len(x)
   return loss

def gradient_descent(x, y_, x_val, y_val, w, lr, numIter, lambda_=0, save_best=True):
   lambda_arr = lambda_ * np.ones(163)
   lambda_arr[-1] = 0

   loss_best = float('inf')
   w_best = np.zeros(163)

   for i in range(numIter):
      train_loss = calculate_loss(x, y_, w)
      val_loss = calculate_loss(x_val, y_val, w)
      if i % 1000 == 0:
         print "Training(Validation) loss after epoch %i: %f(%f)" %(i, train_loss, val_loss)
      w = w - lr*2*(np.dot(x.T, (np.dot(x, w) - y_)) + lambda_arr*w) / len(x)
      if save_best is True and val_loss < loss_best:
         loss_best = val_loss
         w_best = w

   W_ = np.dot(np.linalg.pinv(x), y_)
   pinv_loss = calculate_loss(x_val, y_val, W_)
   print 'pinv loss: ', pinv_loss

   return w_best

def write_file(outfile, y):
   with open(outfile, 'w') as f:
      f.write('id,value\n')
      for i in range(240):
         f.write('id_' + str(i) + ',' + str(y[i]) + '\n')

X, X_val, y_, y_val = split_validation(X, y_, 0.05)
W = gradient_descent(X, y_, X_val, y_val, W, 0.0000015, 200000)
y = np.dot(X_test, W)
val_loss = calculate_loss(X_val, y_val, W)
train_loss = calculate_loss(X, y_, W)
print val_loss
print train_loss
write_file('linear_regression.csv', y)
