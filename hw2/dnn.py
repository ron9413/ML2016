import numpy as np
import pandas as pd
import sys
import cPickle as pickle

nn_input_dim = 57 # input layer dimensionality
nn_output_dim = 1 # output layer dimensionality

#random seed
seed = 259
np.random.seed(seed)

# Adam parameters
lr = 0.01 # learning rate
reg_lambda = 0.25 # regularization strength
epsilon = 1e-8
e = 1e-30
beta1 = 0.9
beta2 = 0.999

def shuffle(X, y_, batch_size):
   indices = np.arange(len(X))
   np.random.shuffle(indices)
   X = X[indices]
   y_ = y_[indices]
   for idx in xrange(0, len(X) / batch_size, batch_size):
      yield X[idx : idx + batch_size], y_[idx : idx + batch_size]

def split_validation(X, y, split):
   indices = np.arange(len(X))
   np.random.shuffle(indices)
   split_idx = int(np.ceil(len(X)*split))
   X_org = X[indices]
   y_org = y[indices]
   X_val = X_org[:split_idx]
   y_val = y_org[:split_idx]
   X = X_org[split_idx:]
   y = y_org[split_idx:]
   return X, X_val, y, y_val

def sigmoid(z):
   return 1 / (1 + np.exp(-z))

def calculate_loss(X, y_, model):
   W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
   # Forward propagation to calculate predictions
   z1 = X.dot(W1) + b1
   a1 = np.maximum(z1, 0)
   z2 = a1.dot(W2) + b2
   a2 = np.maximum(z2, 0)
   z3 = a2.dot(W3) + b3
   y = sigmoid(z3)
   data_loss = np.sum(-(y_ * np.log(y + e) + (1 - y_) * np.log(1 - y + e))) / len(y)
   acc = 1 - np.sum(np.absolute(np.round(y) - y_)) / len(y)
   # Add regulatization term to loss
   #data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
   return data_loss, acc

def predict(model, x):
   W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
   # Forward propagation
   z1 = x.dot(W1) + b1
   a1 = np.maximum(z1, 0)
   z2 = a1.dot(W2) + b2
   a2 = np.maximum(z2, 0)
   z3 = a2.dot(W3) + b3
   y = sigmoid(z3)
   y = np.round(y).astype(int)
   return y.reshape(len(y))

def build_model(X, X_val, y_val, y_, nn_hdim1, nn_hdim2, num_epoch, batch_size, print_loss=True, save_best=True):
   # Initialize the parameters to random values.
   W1 = np.random.normal(0, 1e-6, (nn_input_dim, nn_hdim1))
   b1 = np.zeros((1, nn_hdim1))
   W2 = np.random.normal(0, 1e-6, (nn_hdim1, nn_hdim2))
   b2 = np.zeros((1, nn_hdim2))
   W3 = np.random.normal(0, 1e-6, (nn_hdim2, nn_output_dim))
   b3 = np.zeros((1, nn_output_dim))

   m_W1 = np.zeros((nn_input_dim, nn_hdim1))
   m_b1 = np.zeros((1, nn_hdim1))
   m_W2 = np.zeros((nn_hdim1, nn_hdim2))
   m_b2 = np.zeros((1, nn_hdim2))
   m_W3 = np.zeros((nn_hdim2, nn_output_dim))
   m_b3 = np.zeros((1, nn_output_dim))

   v_W1 = np.zeros((nn_input_dim, nn_hdim1))
   v_b1 = np.zeros((1, nn_hdim1))
   v_W2 = np.zeros((nn_hdim1, nn_hdim2))
   v_b2 = np.zeros((1, nn_hdim2))
   v_W3 = np.zeros((nn_hdim2, nn_output_dim))
   v_b3 = np.zeros((1, nn_output_dim))

   model = {}
   model_best = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3 }

   acc_best = 0
   num_epoch -= 3
   t = 1

   for i in xrange(0, num_epoch):
      for X_sh, y_sh in shuffle(X, y_, batch_size):
         # Forward propagation
         z1 = X_sh.dot(W1) + b1
         a1 = np.maximum(z1, 0)
         z2 = a1.dot(W2) + b2
         a2 = np.maximum(z2, 0)
         z3 = a2.dot(W3) + b3
         y = sigmoid(z3)

         # Backpropagation
         delta4 = y * (1 - y) * (-1) * (y_sh / (y + e) - (1 - y_sh) / (1 - y + e))
         dW3 = (a2.T).dot(delta4)
         db3 = np.sum(delta4, axis=0, keepdims=True)
         delta3 = delta4.dot(W3.T) * (np.greater(a2, 0).astype(float))
         dW2 = (a1.T).dot(delta3)
         db2 = np.sum(delta3, axis=0, keepdims=True)
         delta2 = delta3.dot(W2.T) * (np.greater(a1, 0).astype(float))
         dW1 = np.dot(X_sh.T, delta2)
         db1 = np.sum(delta2, axis=0, keepdims=True)

         # Add regularization terms
         dW3 += reg_lambda * W3
         dW2 += reg_lambda * W2
         dW1 += reg_lambda * W1

         m_W1 = beta1 * m_W1 + (1.0 - beta1) * dW1
         m_b1 = beta1 * m_b1 + (1.0 - beta1) * db1
         m_W2 = beta1 * m_W2 + (1.0 - beta1) * dW2
         m_b2 = beta1 * m_b2 + (1.0 - beta1) * db2
         m_W3 = beta1 * m_W3 + (1.0 - beta1) * dW3
         m_b3 = beta1 * m_b3 + (1.0 - beta1) * db3

         m_W1_hat = m_W1 / (1.0 - beta1**t)
         m_b1_hat = m_b1 / (1.0 - beta1**t)
         m_W2_hat = m_W2 / (1.0 - beta1**t)
         m_b2_hat = m_b2 / (1.0 - beta1**t)
         m_W3_hat = m_W3 / (1.0 - beta1**t)
         m_b3_hat = m_b3 / (1.0 - beta1**t)

         v_W1 = beta2 * v_W1 + (1.0 - beta2) * np.square(dW1)
         v_b1 = beta2 * v_b1 + (1.0 - beta2) * np.square(db1)
         v_W2 = beta2 * v_W2 + (1.0 - beta2) * np.square(dW2)
         v_b2 = beta2 * v_b2 + (1.0 - beta2) * np.square(db2)
         v_W3 = beta2 * v_W3 + (1.0 - beta2) * np.square(dW3)
         v_b3 = beta2 * v_b3 + (1.0 - beta2) * np.square(db3)

         v_W1_hat = v_W1 / (1.0 - beta2**t)
         v_b1_hat = v_b1 / (1.0 - beta2**t)
         v_W2_hat = v_W2 / (1.0 - beta2**t)
         v_b2_hat = v_b2 / (1.0 - beta2**t)
         v_W3_hat = v_W3 / (1.0 - beta2**t)
         v_b3_hat = v_b3 / (1.0 - beta2**t)

         W1 -= lr * m_W1_hat / (np.sqrt(v_W1_hat) + epsilon)
         b1 -= lr * m_b1_hat / (np.sqrt(v_b1_hat) + epsilon)
         W2 -= lr * m_W2_hat / (np.sqrt(v_W2_hat) + epsilon)
         b2 -= lr * m_b2_hat / (np.sqrt(v_b2_hat) + epsilon)
         W3 -= lr * m_W3_hat / (np.sqrt(v_W3_hat) + epsilon)
         b3 -= lr * m_b3_hat / (np.sqrt(v_b3_hat) + epsilon)

         t += 1

      # Assign new parameters to the model
      model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3 }

      # Print the loss.
      if print_loss:
         train_loss, train_acc = calculate_loss(X, y_, model)
         val_loss, val_acc = calculate_loss(X_val, y_val, model)
         if i % 100 == 0:
            print "Training(Validation) loss/acc after epoch %i: %f(%f) %f/%f" \
                  %(i, train_loss, val_loss, train_acc, val_acc)

      # Save best model
      if save_best is True and val_acc > acc_best:
            acc_best = val_acc
            model_best['W1'] = np.copy(W1)
            model_best['b1'] = np.copy(b1)
            model_best['W2'] = np.copy(W2)
            model_best['b2'] = np.copy(b2)
            model_best['W3'] = np.copy(W3)
            model_best['b3'] = np.copy(b3)
   return model_best

def write_file(outfile, y):
   with open(outfile, 'w') as f:
      f.write('id,label\n')
      for i in range(len(y)):
         f.write(str(i+1) + ',' + str(y[i]) + '\n')

if len(sys.argv) == 3:
   training_data = sys.argv[1]
   output_model = sys.argv[2]
   train_data = pd.read_csv(training_data, header=None, usecols=range(1, 59))
   train_arr = np.asarray(train_data)
   X_train = train_arr[:, :-1]
   y_train = train_arr[:, -1]
   y_train = y_train.reshape(len(y_train), 1)

   X, X_val, y_, y_val = split_validation(X_train, y_train, 0.1)
   model = build_model(X, X_val, y_val, y_, 69, 32, 1200, len(X)/6, save_best=False)
   pickle.dump(model, open(output_model, 'wb'))
elif len(sys.argv) == 4:
   model_name = sys.argv[1]
   testing_data = sys.argv[2]
   prediction = sys.argv[3]
   model= pickle.load(open(model_name, 'rb'))
   if prediction.rfind('.csv') == -1:
      prediction = prediction + '.csv'
   test_data = pd.read_csv(testing_data, header=None, usecols=range(1, 58))
   X_test = np.asarray(test_data)
   y = predict(model, X_test)
   write_file(prediction, y)
