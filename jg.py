import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed

FILE = 'jokes.txt'
SEQUENCE_SIZE = 50
EPOCH_NUMBER = 50
HIDDEN_NUMBER = 256
LAYER_NUMBER = 3
BATCH_SIZE = 50

def load(file, sequence_size):
    data = open(file, 'r', encoding = 'utf8').read()
    chars = list(set(data))
    char_to_index = {char : index for index, char in enumerate(chars)}
    x = np.zeros((len(data) // sequence_size, sequence_size, len(chars)))
    y = np.zeros((len(data) // sequence_size, sequence_size, len(chars)))
    for i in range(len(data) // sequence_size):
        x_sequence_index = [char_to_index[j] for j in data[i * sequence_size : (i + 1) * sequence_size]]
        i_s = np.zeros((sequence_size, len(chars)))
        for j in range(sequence_size):
            i_s[j][x_sequence_index[j]] = 1
            x[i] = i_s
        y_sequence_index = [char_to_index[j] for j in data[(i * sequence_size) + 1 : ((i + 1) * sequence_size) + 1]]
        t_s = np.zeros((sequence_size, len(chars)))
        for j in range(sequence_size):
            t_s[j][y_sequence_index[j]] = 1
            y[i] = t_s
    return x, y, len(chars), {index : char for index, char in enumerate(chars)}
def generate(model, length, vocab_size, index_to_char):
    index = [np.random.randint(vocab_size)]
    x = np.zeros((1, length, vocab_size))
    for i in range(length):
        x[0, i, :][index[-1]] = 1
        print(index_to_char[index[-1]], end = '')
        index = np.argmax(model.predict(x[:, : i + 1, :])[0], 1)
x, y, vocab_size, index_to_char = load(FILE, SEQUENCE_SIZE)
model = Sequential()
model.add(LSTM(HIDDEN_NUMBER, input_shape = (None, vocab_size), return_sequences = True))
for i in range(LAYER_NUMBER - 1):
    model.add(LSTM(HIDDEN_NUMBER, return_sequences = True))
model.add(TimeDistributed(Dense(vocab_size)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
for i in range(EPOCH_NUMBER):
    model.fit(x, y, batch_size = BATCH_SIZE, verbose = 1, epochs = 1)
    generate(model, 100, vocab_size, index_to_char)