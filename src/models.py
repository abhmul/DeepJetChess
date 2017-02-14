from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Convolution2D, Reshape
from keras.models import Sequential


def relu3layer(num_channels=12, optimizer=None):
    # Model from https://erikbern.com/2014/11/29/deep-learning-for-chess/
    model = Sequential()
    model.add(Flatten(input_shape=(8, 8, num_channels)))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(64, activation='softmax'))
    model.add(Reshape((8, 8)))

    model.compile(optimizer if optimizer is not None else'rmsprop',
                  'categorical_crossentropy', metrics=['accuracy'])

    return model


def conv10layer(num_channels=12, optimizer=None):
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(8, 8, num_channels), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu'))

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='valid', activation='relu'))

    model.add(Flatten())
    model.add(Dense(64, activation='softmax'))
    model.add(Reshape((8, 8)))

    model.compile(optimizer if optimizer is not None else'rmsprop',
                  'categorical_crossentropy', metrics=['accuracy'])

    return model
