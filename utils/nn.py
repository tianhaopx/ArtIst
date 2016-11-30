from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

# shape(sameple_size,sample_row,sample_column)
def create_lstm_network(samples, features, num_recurrent_units=1):

    model = Sequential()

    #This layer converts frequency space to hidden space
    # http://keras-cn.readthedocs.io/en/latest/layers/core_layer/
    # http://keras-cn.readthedocs.io/en/latest/layers/wrapper/
    model.add(TimeDistributed(input_shape=(num_frequency_vectores,num_frequency_dimensions)))

    for cur_unit in range(num_recurrent_units):
        model.add(LSTM(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))

    #This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model
