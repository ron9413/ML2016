import numpy as np
import pandas as pd
import sys
import cPickle as pickle

np.random.seed(7)

def split_validation(x, y, split):
   indices = np.arange(len(x))
   np.random.shuffle(indices)
   x_org = x[indices]
   y_org = y[indices]
   x_val = x_org[:int(len(x)*split)]
   y_val = y_org[:int(len(y)*split)]
   x = x_org[int(len(x)*split):]
   y = y_org[int(len(y)*split):]
   return x, x_val, y, y_val

def sigmoid(z):
   return 1 / (1 + np.exp(-z))

def calculate_loss(x, y_, w, b):
   y = foward(x, w, b)
   e = 1e-20
   return np.sum(-(y_ * np.log(y + e) + (1 - y_) * np.log(1 - y + e))) / len(y)

def foward(x, w, b):
   z1 = np.dot(x, w) + b
   y = sigmoid(z1)
   return y

def fit(x, y_, x_val, y_val, lr, numEpoch, lambda_=0):
   w = np.zeros(len(x[0]))
   b = 0

   m_w = np.zeros(len(x[0]))
   m_b = 0
   v_w = np.zeros(len(x[0]))
   v_b = 0

   beta1 = 0.9
   beta2 = 0.999
   epsilon = 1e-8
   t = 0

   loss_best = float('inf')
   w_best = np.zeros(len(x[0]))
   b_best = 0
   i_best = 0

   for i in range(numEpoch):
      y = foward(x, w, b)

      #adam
      t += 1
      dw = (np.dot(x.T, y - y_) + lambda_ * w) / len(X)
      db = np.sum(y - y_) / len(x)
      m_w = beta1 * m_w + (1.0 - beta1) * dw
      m_b = beta1 * m_b + (1.0 - beta1) * db
      m_w_hat = m_w / (1.0 - beta1**t)
      m_b_hat = m_b / (1.0 - beta1**t)
      v_w = beta2 * v_w + (1.0 - beta2) * np.square(dw)
      v_b = beta2 * v_b + (1.0 - beta2) * np.square(db)
      v_w_hat = v_w / (1.0 - beta2**t)
      v_b_hat = v_b / (1.0 - beta2**t)
      w -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
      b -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

      train_loss = calculate_loss(x, y_, w, b)
      val_loss = calculate_loss(x_val, y_val, w, b)
      if i % 1000 == 0:
         print 'Training(Validation) loss after %i epoch: %f(%f)' %(i, train_loss, val_loss)
      if val_loss < loss_best:
         loss_best = val_loss
         w_best = np.copy(w)
         b_best = b

   return w_best, b_best

def write_file(outfile, y):
   with open(outfile, 'w') as f:
      f.write('id,label\n')
      for i in range(len(y)):
         if y[i] > 0.5:
            f.write(str(i+1) + ',' + str(1) + '\n')
         else:
            f.write(str(i+1) + ',' + str(0) + '\n')

if len(sys.argv) == 3:
   training_data = sys.argv[1]
   output_model = sys.argv[2]
   train_data = pd.read_csv(training_data, header=None, usecols=range(1, 59))
   train_arr = np.asarray(train_data)
   X_train = train_arr[:, :-1]
   y_train = train_arr[:, -1]

   X, X_val, y_, y_val = split_validation(X_train, y_train, 0.1)
   w, b = fit(X, y_, X_val, y_val, 0.001, 30000)
   model = {'w': w, 'b': b}
   pickle.dump(model, open(output_model, 'wb'))
elif len(sys.argv) == 4:
   model_name = sys.argv[1]
   testing_data = sys.argv[2]
   prediction = sys.argv[3]
   model= pickle.load(open(model_name, 'rb'))
   w, b = model['w'], model['b']
   if prediction.rfind('.csv') == -1:
      prediction = prediction + '.csv'
   test_data = pd.read_csv(testing_data, header=None, usecols=range(1, 58))
   X_test = np.asarray(test_data)
   y_test = foward(X_test, w, b)
   write_file('prediction.csv', y_test)
