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

# Input the model name you want to use
epoch = str(input('the model you want to use:'))
path = os.getcwd()+'/Model_Weight/Weight.'+epoch+'.hdf5'
model = network_utils.new_lstm_network()
# change the path of the weight file to load different weight
model.load_weights(path)
print(model.summary())


# remain to be done
a = X[0].reshape(1,X[0].shape[0],128)

net_output = model.predict(a)


roll = parser.NetOutToPianoRoll(net_output[0])
parser.createMidiFromPianoRoll(roll)
