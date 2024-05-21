from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout, SimpleRNN, GRU


def Bidirectional_RNN_model(input_shape, num_classes):
    model = Sequential()
    model.add(Bidirectional(SimpleRNN(128, input_shape=input_shape, return_sequences=True)))  # Using SimpleRNN instead of LSTM
    model.add(Dropout(0.2))
    model.add(Bidirectional(SimpleRNN(64)) ) # Using SimpleRNN instead of LSTM
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model