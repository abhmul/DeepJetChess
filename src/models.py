from keras.layers import Dense, Flatten, Reshape, Input, merge, AveragePooling2D # noqa D100
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
import keras.backend as K

from functools import partial


def top_k_categorical_accuracy(y_true, y_pred, k=3):
    y_true_flat = K.batch_flatten(y_true)
    y_pred_flat = K.batch_flatten(y_pred)
    return K.mean(K.in_top_k(y_pred_flat, K.argmax(y_true_flat, axis=-1), k), axis=-1)

top_3 = top_k_categorical_accuracy

# 16,84 % val acc, .1206 loss after 4 epochs on 2016
def relu3layer(num_channels=12, prev_boards=0, selection_labels=True, movement_labels=False, optimizer=None, to_load=None):
    # Model from https://erikbern.com/2014/11/29/deep-learning-for-chess/
    input_img = Input(shape=(8, 8, num_channels * (prev_boards + 1)))

    x = Flatten(input_shape=(8, 8, num_channels * (prev_boards + 1)))(input_img)
    x = Dense(2048, activation='relu')(x)
    x = Dense(2048, activation='relu')(x)
    x = Dense(2048, activation='relu')(x)
    selections_flat = Dense(64, activation='softmax')(x)
    selections = Reshape((8, 8))(selections_flat)

    output = [selections]
    model = Model(inputs=input_img, outputs=output)

    if to_load is not None:
        model.load_weights(to_load)

    if movement_labels:
        y = Concatenate()([x, selections_flat])
        y = Dense(256, activation='relu')(y)
        movements = Dense(64, activation='softmax')(y)
        movements = Reshape((8, 8), name='movements')(movements)
        output.append(movements)
        if not selection_labels:
            output = output[1]
        model = Model(inputs=input_img, outputs=output)

    print("Using optimizer: {}".format(optimizer if optimizer is not None else'adam'))

    model.compile(optimizer if optimizer is not None else'adam',
                  'categorical_crossentropy', metrics=['accuracy', top_3])

    return model


def pos2vec(num_channels=12, prev_boards=0, selection_labels=True, movement_labels=False, optimizer=None, to_load=None):
    input_img = Input(shape=(8, 8, num_channels * (prev_boards + 1)))

    x = Flatten(input_shape=(8, 8, num_channels * (prev_boards + 1)))(input_img)
    x = Dense(600, activation='relu')(x)
    x = Dense(400, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    selections_flat = Dense(64, activation='softmax')(x)
    selections = Reshape((8, 8))(selections_flat)

    output = [selections]
    model = Model(inputs=input_img, outputs=output)

    if to_load is not None:
        model.load_weights(to_load)

    if movement_labels:
        y = Concatenate()([x, selections_flat])
        y = Dense(256, activation='relu')(y)
        movements = Dense(64, activation='softmax')(y)
        movements = Reshape((8, 8), name='movements')(movements)
        output.append(movements)
        if not selection_labels:
            output = output[1]
        model = Model(inputs=input_img, outputs=output)

    print("Using optimizer: {}".format(optimizer if optimizer is not None else'adam'))

    model.compile(optimizer if optimizer is not None else'adam',
                  'categorical_crossentropy', metrics=['accuracy', top_3])

    return model

# .276 val_loss and 8.5% val acc on 2016 exclude 5
def convchess_b(num_channels=12, prev_boards=0, selection_labels=True, movement_labels=False, optimizer=None, to_load=None):
    input_img = Input(shape=(8, 8, num_channels * (prev_boards + 1)))
    x = Conv2D(32, (3, 3), padding='valid', activation='relu')(input_img)

    x = Flatten()(x)
    selections_flat = Dense(128, activation='relu')(x)
    selections_flat = Dense(64, activation='softmax')(selections_flat)
    selections = Reshape((8, 8))(selections_flat)

    output = [selections]
    model = Model(inputs=input_img, outputs=output)

    if to_load is not None:
        model.load_weights(to_load)

    if movement_labels:
        y = Concatenate()([x, selections_flat])
        y = Dense(256, activation='relu')(y)
        movements = Dense(64, activation='softmax')(y)
        movements = Reshape((8, 8), name='movements')(movements)
        output.append(movements)
        if not selection_labels:
            output = output[1]
        model = Model(inputs=input_img, outputs=output)

    print("Using optimizer: {}".format(optimizer if optimizer is not None else'adam'))

    model.compile(optimizer if optimizer is not None else'adam',
                  'categorical_crossentropy', metrics=['accuracy', top_3])

    return model

# Scores 25% val acc and .0627 (going down 7 ep) val_loss on cvc 2015 exclude5 prev4
def conv4layer(num_channels=12, prev_boards=0, selection_labels=True, movement_labels=False, optimizer=None, to_load=None):
    input_img = Input(shape=(8, 8, num_channels * (prev_boards + 1)))

    x = Conv2D(64, (3, 3), padding='valid', activation='relu')(input_img)
    x = Conv2D(128, (3, 3), padding='valid', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='valid', activation='relu')(x)
    x = Conv2D(256, (2, 2), padding='valid', activation='relu')(x)

    x = Flatten()(x)
    selections_flat = Dense(64, activation='softmax')(x)
    selections = Reshape((8, 8), name='selections')(selections_flat)
    output = [selections]
    model = Model(inputs=input_img, outputs=output)

    if to_load is not None:
        model.load_weights(to_load)

    if movement_labels:
        y = Concatenate()([x, selections_flat])
        y = Dense(256, activation='relu')(y)
        movements = Dense(64, activation='softmax')(y)
        movements = Reshape((8, 8), name='movements')(movements)
        output.append(movements)
        if not selection_labels:
            output = output[1]
        model = Model(inputs=input_img, outputs=output)

    print("Using optimizer: {}".format(optimizer if optimizer is not None else'adam'))

    model.compile(optimizer if optimizer is not None else'adam',
                  'categorical_crossentropy', metrics=['accuracy', top_3])

    return model


# Scores 22% val acc on small 2016
def conv10layer(num_channels=12, prev_boards=0, selection_labels=True, movement_labels=False, optimizer=None, to_load=None):
    input_img = Input(shape=(8, 8, num_channels * (prev_boards + 1)))

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='valid', activation='relu')(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='valid', activation='relu')(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='valid', activation='relu')(x)

    x = Flatten()(x)
    selections_flat = Dense(64, activation='softmax')(x)
    selections = Reshape((8, 8))(selections_flat)

    output = [selections]
    model = Model(inputs=input_img, outputs=output)

    if to_load is not None:
        model.load_weights(to_load)

    if movement_labels:
        y = Concatenate()([x, selections_flat])
        y = Dense(256, activation='relu')(y)
        movements = Dense(64, activation='softmax')(y)
        movements = Reshape((8, 8), name='movements')(movements)
        output.append(movements)
        if not selection_labels:
            output = output[1]
        model = Model(inputs=input_img, outputs=output)

    print("Using optimizer: {}".format(optimizer if optimizer is not None else'adam'))

    model.compile(optimizer if optimizer is not None else'adam',
                  'categorical_crossentropy', metrics=['accuracy', top_3])

    return model

# 18.9 % accuracy .1106 loss w/ 2016 exclude5
def conv_searcher(num_channels=12, prev_boards=0, selection_labels=True, movement_labels=False, optimizer=None, to_load=None):
    input_img = Input(shape=(8, 8, num_channels * (prev_boards + 1)))

    # Local searcher
    local = Conv2D(64, (3, 3), padding='valid', activation='relu')(input_img)
    local = Conv2D(128, (3, 3), padding='valid', activation='relu')(local)
    local = Conv2D(256, (3, 3), padding='valid', activation='relu')(local)
    local = Conv2D(512, (2, 2), padding='valid', activation='relu')(local)
    # Creates a 1 x 1 x 512 output

    # Column searcher
    column = Conv2D(64, (8, 1), padding='valid', activation='relu')(input_img) # 1 x 8 x 64
    column = Conv2D(128, (1, 3), padding='valid', activation='relu')(column) # 1 x 6 x 128
    column = Conv2D(256, (1, 3), padding='valid', activation='relu')(column) # 1 x 4 x 256
    column = Conv2D(512, (1, 3), padding='valid', activation='relu')(column) # 1 x 2 x 512
    column = Conv2D(512, (1, 2), padding='valid', activation='relu')(column) # 1 x 1 x 512

    # Row searcher
    row = Conv2D(64, (1, 8), padding='valid', activation='relu')(input_img) # 8 x 1 x 64
    row = Conv2D(128, (3, 1), padding='valid', activation='relu')(row) # 6 x 1 x 128
    row = Conv2D(256, (3, 1), padding='valid', activation='relu')(row) # 4 x 1 x 256
    row = Conv2D(512, (3, 1), padding='valid', activation='relu')(row) # 2 x 1 x 512
    row = Conv2D(512, (2, 1), padding='valid', activation='relu')(row) # 1 x 1 x 512

    # Merge the local, row, and column searchers
    searcher = merge([local, row, column], mode='concat', concat_axis=3)
    searcher = Flatten()(searcher)

    # Create the global searcher
    glob = Flatten()(input_img)
    glob = Dense(1024, activation='relu')(glob)

    # Merge the conv searchers with the global
    x = merge([glob, searcher], mode='concat', concat_axis=1)
    selections_flat = Dense(64, activation='softmax')(x)
    selections = Reshape((8, 8))(selections_flat)
    output = [selections]
    model = Model(inputs=input_img, outputs=output)

    if to_load is not None:
        model.load_weights(to_load)

    if movement_labels:
        y = Concatenate()([x, selections_flat])
        y = Dense(256, activation='relu')(y)
        movements = Dense(64, activation='softmax')(y)
        movements = Reshape((8, 8), name='movements')(movements)
        output.append(movements)
        if not selection_labels:
            output = output[1]
        model = Model(inputs=input_img, outputs=output)

    print("Using optimizer: {}".format(optimizer if optimizer is not None else'adam'))

    model.compile(optimizer if optimizer is not None else'adam',
                  'categorical_crossentropy', metrics=['accuracy', top_3])

    return model




def conv16layer(num_channels=12, prev_boards=0, selection_labels=True, movement_labels=False, optimizer=None, to_load=None):

    input_img = Input(shape=(8, 8, num_channels * (prev_boards + 1)))

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='valid', activation='relu')(x)
    # model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='valid', activation='relu')(x)
    # model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='valid', activation='relu')(x)
    # model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (2, 2), padding='valid', activation='relu')(x)
    # model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

    x = Flatten()(x)
    selections_flat = Dense(64, activation='softmax')(x)
    selections = Reshape((8, 8))(selections_flat)

    output = [selections]
    model = Model(inputs=input_img, outputs=output)

    if to_load is not None:
        model.load_weights(to_load)

    if movement_labels:
        y = Concatenate()([x, selections_flat])
        y = Dense(256, activation='relu')(y)
        movements = Dense(64, activation='softmax')(y)
        movements = Reshape((8, 8), name='movements')(movements)
        output.append(movements)
        if not selection_labels:
            output = output[1]
        model = Model(inputs=input_img, outputs=output)

    print("Using optimizer: {}".format(optimizer if optimizer is not None else'adam'))

    model.compile(optimizer if optimizer is not None else'adam',
                  'categorical_crossentropy', metrics=['accuracy', top_3])

    return model


def modified_inception_module(input_img, num_filters):

    tower_1 = Conv2D(num_filters, 1, 1, padding='same', activation='relu')(input_img)
    tower_1 = Conv2D(num_filters, 3, 3, padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(num_filters, 1, 1, padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(num_filters, 5, 5, padding='same', activation='relu')(tower_2)

    tower_3 = Conv2D(num_filters, 1, 1, padding='same', activation='relu')(input_img)
    # tower_1 = Conv2D(num_filters, 3, 3, padding='same', activation='relu')(tower_3)

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
