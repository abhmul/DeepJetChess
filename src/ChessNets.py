from keras.models import Sequential
from keras.layers import Dense

def baseline():

    # Model from https://erikbern.com/2014/11/29/deep-learning-for-chess/
    model = Sequential()
    model.add(Dense(2048, activation='relu', input_dim=8*8*12))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(64, activation='softmax'))

    model.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])

    return model

