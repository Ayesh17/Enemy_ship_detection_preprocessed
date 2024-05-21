from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout
from keras.layers import LSTM, Dense, Dropout


def RNN_model(input_shape, num_classes):
    model = Sequential()
    model.add(SimpleRNN(128, input_shape=input_shape, return_sequences=True))  # Using SimpleRNN instead of LSTM
    model.add(Dropout(0.2))
    model.add(SimpleRNN(64))  # Using SimpleRNN instead of LSTM
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model