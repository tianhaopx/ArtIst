import numpy as np
import os
from utils import nn as network_utils


#Load training data
print('Loading training data')
inputFile = 'out_put'
X_train = np.load('data_x.npy')
y = np.load('data_y.npy')
print('Finished loading training data')

model = network_utils.create_lstm_network()


# the parameter here can be changed
model.fit(X_train, y, batch_size=64,
          nb_epoch=10, verbose=1,
          validation_split=0.0)

model.save_weights('test_3d_model')
