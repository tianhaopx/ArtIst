from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

# 2000 and 4
# 2000 is the hidden layers and 4 is the dimension of our own data
# these two numbers are really important
def create_lstm_network():
    model = Sequential()
    # This layer converts frequency space to hidden space
    model.add(LSTM(256, input_dim=128,return_sequences=True))
    for cur_unit in range(2):
        model.add(LSTM(256, return_sequences=True))
    # This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(input_dim=256,output_dim=128)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    return model

# Add the drop out layer
# this lstm nn is proved can work
def new_lstm_network(input_dim, output_dim):
    model = Sequential()
    model.add(LSTM(input_dim=input_dim, output_dim=output_dim, activation='sigmoid', return_sequences=False))
    model.add(Dropout(0.2))
    #model.add(TimeDistributed(Dense(1)))
    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary', metrics=['accuracy'])
    return model
