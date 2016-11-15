import numpy as np
import os
from utils import nn as network_utils

cur_iter = 0
model_basename = 'Test_train'
model_filename = model_basename + str(cur_iter)

#Load training data
print('Loading training data')
inputFile = 'test'
X_train = np.load(inputFile + '_x.npy')
y_train = np.load(inputFile + '_y.npy')
print('Finished loading training data')

#Figure out how many frequencies we have in the data
freq_space_vec = X_train.shape[1]
freq_space_dims = X_train.shape[2]
hidden_dims = 1024

#Creates a lstm network
model = network_utils.create_lstm_network(num_frequency_vectores=freq_space_vec, num_frequency_dimensions=freq_space_dims, num_hidden_dimensions=hidden_dims)
#You could also substitute this with a RNN or GRU
#model = network_utils.create_gru_network()

#Load existing weights if available
if os.path.isfile(model_filename):
    model.load_weights(model_filename)

num_iters = 1000          #Number of iterations for training
epochs_per_iter = 100    #Number of iterations before we save our model
batch_size = 100          #Number of training examples pushed to the GPU per batch.
                        #Larger batch sizes require more memory, but training will be faster
print('Starting training!')
while cur_iter < num_iters:
    print('Iteration: ' + str(cur_iter))
    #We set cross-validation to 0,
    #as cross-validation will be on different datasets
    #if we reload our model between runs
    #The moral way to handle this is to manually split
    #your data into two sets and run cross-validation after
    #you've trained the model for some number of epochs
    history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs_per_iter, verbose=1, validation_split=0.0)
    cur_iter += epochs_per_iter
    model.save_weights(model_basename + str(cur_iter))
print ('Training complete!')
model.save_weights(model_basename + str(cur_iter))

