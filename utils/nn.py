from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

# 2000 and 4
# 2000 is the hidden layers and 4 is the dimension of our own data
# these two numbers are really important
def create_lstm_network():
    model = Sequential()
    # This layer converts frequency space to hidden space
    model.add(LSTM(2000, input_dim=4,return_sequences=True))
    for cur_unit in range(2):
        model.add(LSTM(2000, return_sequences=True))
    # This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(input_dim=2000,output_dim=4)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    return model
