from keras.layers import Dense, Flatten, Reshape, Input, merge, AveragePooling2D, concatenate # noqa D100
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
import keras.backend as K

def modified_inception_module(input_img, num_filters):

    tower_1 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_img)
    tower_1 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(num_filters, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_img)
    # tower_1 = Conv2D(num_filters, 3, 3, padding='same', activation='relu')(tower_3)

    output = merge([tower_1, tower_2, tower_3], mode='concat', concat_axis=3)
    return output


# Towers
def pos2vec():
    input_vec = Input(shape=(769,))

    x = Dense(600, activation='relu')(input_vec)
    x = Dense(400, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    output = Dense(100, activation='relu')(x)

    return Model(inputs=input_vec, outputs=output)

def conv4layer_tower():
    input_img = Input(shape=(8, 8, 13))

    x = Conv2D(64, (3, 3), padding='valid', activation='relu')(input_img)
    x = Conv2D(128, (3, 3), padding='valid', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='valid', activation='relu')(x)
    x = Conv2D(256, (2, 2), padding='valid', activation='relu')(x)
    output = Flatten()(x)

    return Model(inputs=input_img, outputs=output)

def conv10layer_tower():
    input_img = Input(shape=(8, 8, 13))

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='valid', activation='relu')(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='valid', activation='relu')(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='valid', activation='relu')(x)

    output = Flatten()(x)

    return Model(inputs=input_img, outputs=output)

def inception_tower(filters=[32, 64, 128, 256, 256]):
    input_img = Input(shape=(8, 8, 13))

    x = input_img
    for nfilt in filters:
        x = modified_inception_module(x, nfilt)

    output = Flatten()(x)

    return Model(inputs=input_img, outputs=output)

def conv_searcher_tower():
    input_img = Input(shape=(8, 8, 13))

    # Local searcher
    local = Conv2D(64, (3, 3), padding='valid', activation='relu')(input_img)
    local = Conv2D(128, (3, 3), padding='valid', activation='relu')(local)
    local = Conv2D(256, (3, 3), padding='valid', activation='relu')(local)
    local = Conv2D(512, (2, 2), padding='valid', activation='relu')(local) # 1 x 1 x 512
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
    searcher = concatenate([local, row, column], axis=-1)
    searcher = Flatten()(searcher)

    # Create the global searcher
    glob = Flatten()(input_img)
    glob = Dense(1024, activation='relu')(glob)
    glob = Dense(512, activation='relu')(glob)
    glob = Dense(256, activation='relu')(glob)
    glob = Dense(128, activation='relu')(glob)

    x = concatenate([glob, searcher], axis=-1) # 128 + 512*3 = 1564
    output = Dense(512, activation='relu')(x)

    return Model(inputs=input_img, outputs=output)

# Full models
def deepchess(optimizer=None):

    b1 = Input(shape=(769,))
    b2 = Input(shape=(769,))

    tower = pos2vec()

    w = tower(b1)
    x = tower(b2)

    merged_vector = concatenate([w, x], axis=-1)
    combined = Dense(400, activation='relu')(merged_vector)
    combined = Dense(200, activation='relu')(combined)
    combined = Dense(100, activation='relu')(combined)
    output = Dense(2, activation='softmax')(merged_vector)

    model = Model(inputs=[b1, b2], outputs=output)
    model.compile(optimizer if optimizer is not None else 'adam',
                  'categorical_crossentropy', metrics=['accuracy'])
    return model

def conv4layer(optimizer=None):

    b1 = Input(shape=(8, 8, 13))
    b2 = Input(shape=(8, 8, 13))

    tower = conv4layer_tower()

    w = tower(b1)
    x = tower(b2)

    merged_vector = concatenate([w, x], axis=-1)
    z = Dense(128, activation='relu')(merged_vector)
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[b1, b2], outputs=output)
    model.compile(optimizer if optimizer is not None else'adam',
                  'binary_crossentropy', metrics=['accuracy'])
    return model

def comparator(tower_func, optimizer=None):

    b1 = Input(shape=(8, 8, 13))
    b2 = Input(shape=(8, 8, 13))

    tower = tower_func()

    w = tower(b1)
    x = tower(b2)

    merged_vector = concatenate([w, x], axis=-1)
    output = Dense(1, activation='sigmoid')(merged_vector)

    model = Model(inputs=[b1, b2], outputs=output)
    model.compile(optimizer if optimizer is not None else 'adam',
                  'binary_crossentropy', metrics=['accuracy'])
    return model

# Ep43
# loss: 0.1846 - acc: 0.9205
# val_loss: 0.1824 - val_acc: 0.9217
def conv10layer(optimizer=None):

    return comparator(conv10layer_tower, optimizer=optimizer)

#Ep 28
def incpetion_net(optimizer=None):

    return comparator(inception_tower, optimizer=optimizer)

def conv_searcher(optimizer=None):

    return comparator(conv_searcher_tower, optimizer=optimizer)
