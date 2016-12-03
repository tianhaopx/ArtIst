import numpy as np
import os
from utils import nn as network_utils
from utils import parser

#Load training data
print('Loading training data')
inputFile = 'out_put'
X = np.load('data_x.npy')
y = np.load('data_y.npy')
print('Finished loading training data')

input_dim = X.shape[2]
output_dim = y.shape[1]
epoch = str(input('the model you want to use:'))
path = os.getcwd()+'/Model_Weight/Weight.'+epoch+'.hdf5'
model = network_utils.new_lstm_network(input_dim, output_dim)
# change the path of the weight file to load different weight
model.load_weights(path)


net_output = model.predict(X)
roll = parser.NetOutToPianoRoll(net_output)
parser.createMidiFromPianoRoll(roll)
