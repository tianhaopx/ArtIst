import numpy as np
import os
from utils import nn as network_utils


#Load training data
print('Loading training data')
inputFile = 'out_put'
X = np.load('data_x.npy')
y = np.load('data_y.npy')
print('Finished loading training data')

input_dim = X.shape[2]
output_dim = y.shape[1]


model = network_utils.new_lstm_network(input_dim, output_dim)

print(X.shape,y.shape)
# the parameter here can be changed
model.fit(X, y, batch_size=128,
          nb_epoch=10, verbose=1,
          validation_split=0.0)

model.save_weights('test_3d_model')
