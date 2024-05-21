from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.layers import LSTM, Dense, Dropout


def GRU_model(input_shape, num_classes):
    model = Sequential()
    model.add(GRU(128, input_shape=input_shape, return_sequences=True))  # Using GRU instead of SimpleRNN
    model.add(Dropout(0.2))
    model.add(GRU(64))  # Using GRU instead of SimpleRNN
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model