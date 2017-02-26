from keras.layers import Dense, Flatten, Convolution2D, Reshape, Input, merge, AveragePooling2D # noqa D100
from keras.models import Sequential, Model

# 16,84 % val acc, .1206 loss after 4 epochs on 2016
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

# .276 val_loss and 8.5% val acc on 2016 exclude 5
def convchess_b(num_channels=12, optimizer=None):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='valid', activation='relu', input_shape=(8, 8, num_channels)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='softmax'))
    model.add(Reshape((8, 8)))

    model.compile(optimizer if optimizer is not None else'rmsprop',
                  'categorical_crossentropy', metrics=['accuracy'])

    return model



# Scores 23.95% val acc on small 2016 with exclude5
# Scores 23 % val acc on large 2016 with exclude5 .105 val loss
# Scores 19.5% val acc and .0807 val_loss on cvc 2026 exclude5
def conv4layer(num_channels=12, selection_labels=True, movement_labels=False, optimizer=None, to_load=None):
    input_img = Input(shape=(8, 8, num_channels))

    x = Convolution2D(64, 3, 3, border_mode='valid', activation='relu', input_shape=(8, 8, num_channels))(input_img)
    x = Convolution2D(128, 3, 3, border_mode='valid', activation='relu')(x)
    x = Convolution2D(256, 3, 3, border_mode='valid', activation='relu')(x)
    x = Convolution2D(256, 2, 2, border_mode='valid', activation='relu')(x)

    x = Flatten()(x)
    selections = Dense(64, activation='softmax')(x)
    selections = Reshape((8, 8), name='selections')(selections)
    output = [selections]
    model = Model(input=input_img, output=output)
    if to_load is not None:
        model.load_weights(to_load)

    if movement_labels:
        movements = Dense(64, activation='softmax')(x)
        movements = Reshape((8, 8), name='movements')(movements)
        output.append(movements)
        if not selection_labels:
            output = output[1]
        model = Model(input=input_img, output=output)


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

# 18.9 % accuracy .1106 loss w/ 2016 exclude5
def conv_searcher(num_channels=12, movements=False, optimizer=None):
    input_img = Input(shape=(8, 8, num_channels))

    # Local searcher
    local = Convolution2D(64, 3, 3, border_mode='valid', activation='relu')(input_img)
    local = Convolution2D(128, 3, 3, border_mode='valid', activation='relu')(local)
    local = Convolution2D(256, 3, 3, border_mode='valid', activation='relu')(local)
    local = Convolution2D(512, 2, 2, border_mode='valid', activation='relu')(local)
    # Creates a 1 x 1 x 512 output

    # Column searcher
    column = Convolution2D(64, 8, 1, border_mode='valid', activation='relu')(input_img) # 1 x 8 x 64
    column = Convolution2D(128, 1, 3, border_mode='valid', activation='relu')(column) # 1 x 6 x 128
    column = Convolution2D(256, 1, 3, border_mode='valid', activation='relu')(column) # 1 x 4 x 256
    column = Convolution2D(512, 1, 3, border_mode='valid', activation='relu')(column) # 1 x 2 x 512
    column = Convolution2D(512, 1, 2, border_mode='valid', activation='relu')(column) # 1 x 1 x 512

    # Row searcher
    row = Convolution2D(64, 1, 8, border_mode='valid', activation='relu')(input_img) # 8 x 1 x 64
    row = Convolution2D(128, 3, 1, border_mode='valid', activation='relu')(row) # 6 x 1 x 128
    row = Convolution2D(256, 3, 1, border_mode='valid', activation='relu')(row) # 4 x 1 x 256
    row = Convolution2D(512, 3, 1, border_mode='valid', activation='relu')(row) # 2 x 1 x 512
    row = Convolution2D(512, 2, 1, border_mode='valid', activation='relu')(row) # 1 x 1 x 512

    # Merge the local, row, and column searchers
    searcher = merge([local, row, column], mode='concat', concat_axis=3)
    searcher = Flatten()(searcher)

    # Create the global searcher
    glob = Flatten()(input_img)
    glob = Dense(1024, activation='relu')(glob)

    # Merge the conv searchers with the global
    output = merge([glob, searcher], mode='concat', concat_axis=1)
    selections = Dense(64, activation='softmax')(x)
    selections = Reshape((8, 8))(selections)
    output = [selections]

    if movements:
        movements = Dense(64, activation='softmax')(x)
        movements = Reshape((8, 8))(movements)
        output.append(movements)


    model = Model(input=input_img, output=output)
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
