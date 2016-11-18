import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dropout
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, model_from_json
import cPickle as pickle
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras import backend as K
import sys

K.set_image_dim_ordering('tf')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

data_path = sys.argv[1]
encoder_name = sys.argv[2]
dnn_name = sys.argv[3]

def auto_encoder(x_train, x_val, num_epoch):
   input_img = Input(shape=(32, 32, 3))

   x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
   x = MaxPooling2D((2, 2), border_mode='same')(x)
   x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
   x = MaxPooling2D((2, 2), border_mode='same')(x)
   x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
   encoded = MaxPooling2D((2, 2), border_mode='same')(x)

   x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
   x = UpSampling2D((2, 2))(x)
   x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
   x = UpSampling2D((2, 2))(x)
   x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
   x = UpSampling2D((2, 2))(x)
   decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

   encoder = Model(input_img, encoded)
   autoencoder = Model(input_img, decoded)
   autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

   autoencoder.fit(x_train, x_train,
                   nb_epoch=num_epoch,
                   batch_size=128,
                   shuffle=True,
                   validation_data=(x_val, x_val))
   encoder.save(encoder_name+'.h5')
   return encoder

def unpickle(filename):
   fo = open(filename, 'rb')
   data = pickle.load(fo)
   fo.close()
   return data

def load_label(data, split=0.1):
   data_arr = np.asarray(data)
   nb_train_samples = int(len(data_arr[0])*(1-split))
   nb_val_samples = int(len(data_arr[0])*split)

   for i in range(len(data_arr)):
      indices = np.arange(len(data_arr[i]))
      np.random.shuffle(indices)
      data_arr[i] = data_arr[i][indices]

   X_train = data_arr[0][:nb_train_samples].reshape(nb_train_samples, 3, 32, 32).transpose(0, 2, 3, 1)
   y_train = np.zeros((nb_train_samples,), dtype=np.int8)
   X_val = data_arr[0][nb_train_samples:].reshape(nb_val_samples, 3, 32, 32).transpose(0, 2, 3, 1)
   y_val = np.zeros((nb_val_samples,), dtype=np.int8)

   for i in range(1, len(data_arr)):
      train_reshape = data_arr[i][:nb_train_samples].reshape(nb_train_samples, 3, 32, 32).transpose(0, 2, 3, 1)
      val_reshape = data_arr[i][nb_train_samples:].reshape(nb_val_samples, 3, 32, 32).transpose(0, 2, 3, 1)
      X_train = np.concatenate((X_train, train_reshape))
      y_train = np.concatenate((y_train, i*np.ones((nb_train_samples,), dtype=np.int8)))
      X_val = np.concatenate((X_val, val_reshape))
      y_val = np.concatenate((y_val, i*np.ones((nb_val_samples,), dtype=np.int8)))

   y_train = np_utils.to_categorical(y_train, 10).astype(np.float32)
   y_val = np_utils.to_categorical(y_val, 10).astype(np.float32)

   X_train = X_train.astype(np.float32)
   X_val = X_val.astype(np.float32)
   X_train /= 255.
   X_val /= 255.

   return X_train, y_train, X_val, y_val

def load_unlabel(data):
   X_unlabel = np.asarray(data).reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)
   X_unlabel = X_unlabel.astype(np.float32)
   X_unlabel /= 255.

   return X_unlabel

def load_test(data):
   X_test = np.asarray(data['data']).reshape(len(data['data']), 3, 32, 32).transpose(0, 2, 3, 1)
   X_test = X_test.astype(np.float32)
   X_test /= 255.

   return X_test

def dnn():
   model = Sequential()
   model.add(Dense(256, input_dim=128))
   model.add(BatchNormalization())
   model.add(Activation('elu'))
   model.add(Dropout(0.25))
   model.add(Dense(512))
   model.add(BatchNormalization())
   model.add(Activation('elu'))
   model.add(Dropout(0.25))
   model.add(Dense(32))
   model.add(BatchNormalization())
   model.add(Activation('elu'))
   model.add(Dropout(0.5))
   model.add(Dense(10))
   model.add(Activation('softmax'))

   return model

def supervised_learning(X_train, y_train, num_epoch, weights=None):
   model = model_from_json(open(dnn_name+'.json').read())
   if weights:
      model.load_weights(weights)

   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   model.summary()
   callbacks = [EarlyStopping(monitor='val_acc', patience=num_epoch/2), mc]

   model.fit(X_train, y_train, batch_size=32, nb_epoch=num_epoch,
             callbacks=callbacks, validation_data=(X_val, y_val))

def self_learning(X_add, y_add, X_unlabel, num_epoch):
   model = model_from_json(open(dnn_name+'.json').read())
   model.load_weights(dnn_name+'_weights.h5')
   predict_arr = model.predict(X_unlabel)
   thres = 0.95
   if len(X_add) > 10000:
      thres = 0.9
   confident_data = np.argwhere(predict_arr > thres)
   if len(confident_data) != 0:
      X_add = np.concatenate((X_add, X_unlabel[confident_data[:, 0]]))
      y_add = np.concatenate((y_add, np_utils.to_categorical(confident_data[:, 1], 10)))
      X_unlabel = np.delete(X_unlabel, confident_data[:, 0], axis=0)

   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

   model.fit(X_add, y_add, batch_size=32, nb_epoch=num_epoch,
             callbacks=[mc], validation_data=(X_val, y_val))

   return X_add, y_add, X_unlabel

if __name__ == "__main__":
   all_label = unpickle(data_path+'all_label.p')
   all_unlabel = unpickle(data_path+'all_unlabel.p')
   test = unpickle(data_path+'test.p')
   X_train, y_train, X_val, y_val = load_label(all_label)
   X_unlabel = load_unlabel(all_unlabel)
   X_test = load_test(test)
   X_unlabel = np.concatenate((X_unlabel, X_test))
   X_train_ae = np.concatenate((X_train, X_unlabel))

   encoder = auto_encoder(X_train_ae, X_val, 30)
   X_train = encoder.predict(X_train).reshape(len(X_train), 128)
   X_val = encoder.predict(X_val).reshape(len(X_val), 128)
   X_unlabel = encoder.predict(X_unlabel).reshape(len(X_unlabel), 128)
   X_test = encoder.predict(X_test).reshape(len(X_test), 128)

   X_add = X_train
   y_add = y_train

   mc = ModelCheckpoint(filepath=dnn_name+"_weights.h5", monitor='val_acc', save_best_only=True, verbose=0)

   model = dnn()
   open(dnn_name+'.json', 'w').write(model.to_json())
   weights = None
   num_iter = 5
   train_epoch = 100
   add_epoch = 10
   for i in range(num_iter):
      supervised_learning(X_train, y_train, train_epoch, weights)
      X_add, y_add, X_unlabel = self_learning(X_add, y_add, X_unlabel, add_epoch)
      weights = dnn_name+'_weights.h5'
   supervised_learning(X_train, y_train, train_epoch, weights)

   model = model_from_json(open(dnn_name+'.json').read())
   model.load_weights(dnn_name+'_weights.h5')
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   print model.evaluate(X_val, y_val)[1]

   if tensorflow_backend._SESSION:
      tf.reset_default_graph()
      tensorflow_backend._SESSION.close()
      tensorflow_backend._SESSION = None

