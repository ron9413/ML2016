import numpy as np
from keras.models import model_from_json, load_model
import cPickle as pickle
from keras import backend as K
import sys

K.set_image_dim_ordering('tf')

data_path = sys.argv[1]
encoder_name = sys.argv[2]
dnn_name = sys.argv[3]
predict_csv = sys.argv[4]

def unpickle(filename):
   fo = open(filename, 'rb')
   data = pickle.load(fo)
   fo.close()
   return data

def load_test(data):
   X_test = np.asarray(data['data']).reshape(len(data['data']), 3, 32, 32).transpose(0, 2, 3, 1)
   X_test = X_test.astype(np.float32)
   X_test /= 255.

   return X_test

def encode_img(data):
   encoder = load_model(encoder_name+'.h5')
   X_test = encoder.predict(data).reshape(len(data), 128)

   return X_test

def write_file(outfile, prediction):
   with open(outfile, 'w') as f:
      f.write('ID,class\n')
      for i in range(len(prediction)):
         f.write(str(i) + ',' + str(prediction[i]) + '\n')

if __name__ == "__main__":
   test = unpickle(data_path+'test.p')
   X_test = load_test(test)
   X_test = encode_img(X_test)

   model = model_from_json(open(dnn_name+'.json').read())
   model.load_weights(dnn_name+'_weights.h5')
   prediction = model.predict_classes(X_test, verbose=0)
   write_file(predict_csv, prediction)