from keras.layers import Dense, Flatten, Convolution2D, Reshape, Input, merge, AveragePooling2D # noqa D100
from keras.models import Sequential, Model


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

# Scores 23.95% val acc on small 2016 with exclude5
def conv4layer(num_channels=12, optimizer=None):
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', input_shape=(8, 8, num_channels)))
    model.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='valid', activation='relu'))
    model.add(Convolution2D(256, 2, 2, border_mode='valid', activation='relu'))

    model.add(Flatten())
    model.add(Dense(64, activation='softmax'))
    model.add(Reshape((8, 8)))

    model.compile(optimizer if optimizer is not None else'rmsprop',
                  'categorical_crossentropy', metrics=['accuracy'])

    return model


# Scores 22% val acc on small 2016
def conv10layer(num_channels=12, optimizer=None):
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(8, 8,
                            num_channels), activation='relu'))
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

def conv16layer(num_channels=12, optimizer=None):
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(8, 8,
                            num_channels), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(64, activation='softmax'))
    model.add(Reshape((8, 8)))

    model.compile(optimizer if optimizer is not None else'rmsprop',
                  'categorical_crossentropy', metrics=['accuracy'])

    return model


def modified_inception_module(input_img, num_filters):

    tower_1 = Convolution2D(num_filters, 1, 1, border_mode='same', activation='relu')(input_img)
    tower_1 = Convolution2D(num_filters, 3, 3, border_mode='same', activation='relu')(tower_1)

    tower_2 = Convolution2D(num_filters, 1, 1, border_mode='same', activation='relu')(input_img)
    tower_2 = Convolution2D(num_filters, 5, 5, border_mode='same', activation='relu')(tower_2)

    tower_3 = Convolution2D(num_filters, 1, 1, border_mode='same', activation='relu')(input_img)
    # tower_1 = Convolution2D(num_filters, 3, 3, border_mode='same', activation='relu')(tower_3)

    output = merge([tower_1, tower_2, tower_3], mode='concat', concat_axis=3)
    return output

# Scores 17% Val acc w/ Filters [32, 64, 128, 256, 256] on small 2016
def inception_nlayer(filters, num_channels=12, optimizer=None):

    input_img = Input(shape=(8, 8, num_channels))
    x = input_img
    for nfilt in filters:
        x = modified_inception_module(x, nfilt)

    x = Flatten()(x)
    x = Dense(64, activation='softmax')(x)
    output = Reshape((8, 8))(x)

    model = Model(input=input_img, output=output)
    model.compile(optimizer if optimizer is not None else'rmsprop',
                  'categorical_crossentropy', metrics=['accuracy'])
    return model
