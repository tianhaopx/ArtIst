import numpy as np
import os
from utils import nn as network_utils
from keras.callbacks import ModelCheckpoint


#Load training data
print('Loading training data')
inputFile = 'out_put'
X = np.load('data_x.npy')
y = np.load('data_y.npy')
print('Finished loading training data')

input_dim = X.shape[2]
output_dim = y.shape[1]

save_path = os.getcwd()+'/Model_Weight/Weight.{epoch:02d}.hdf5'
#print(save_path)

model = network_utils.new_lstm_network(input_dim, output_dim)
# You can use this line to see what the model lokks like
# model.summary()

# save our model every epoch
saver = ModelCheckpoint(save_path,
                        save_best_only = False, save_weights_only = True,
                        mode = 'auto')

# the parameter here can be changed due to the differernt setting of the PC
model.fit(X, y,
          batch_size=128, nb_epoch=100, verbose=1,
          validation_split=0.0, callbacks=[saver])
# Save the final model Weight
# This line is useless since we get saver above
# model.save_weights('Fianl_test_model')
