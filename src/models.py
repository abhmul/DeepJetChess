from keras.layers import Dense, Flatten, Reshape, Input, merge, AveragePooling2D, concatenate, ZeroPadding2D, BatchNormalization, Activation # noqa D100
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
import keras.backend as K


# Towers
def cascade_net(num_filters=128, embedding_size=128, include_rights=True):
    input_img = Input(shape=(8, 8, 17)) if include_rights else Input(shape=(8, 8, 13))

    # Tower 1
    tower_1 = Conv2D(num_filters, (1, 1), padding='valid', activation='relu')(input_img)
    tower_1 = Conv2D(num_filters, (3, 3), padding='valid', activation='relu')(tower_1)
    tower_1 = Conv2D(num_filters, (1, 1), padding='valid', activation='relu')(tower_1)
    tower_1 = Conv2D(num_filters, (3, 3), padding='valid', activation='relu')(tower_1)

    # Tower 2
    tower_2 = Conv2D(num_filters, (1, 1), padding='valid', activation='relu')(input_img)
    tower_2 = Conv2D(num_filters, (5, 5), padding='valid', activation='relu')(tower_2)

    # Merge Tower1 and Tower2
    tower_12 = concatenate([tower_1, tower_2], axis=-1)
    tower_12 = Conv2D(num_filters*2, (1, 1), padding='valid', activation='relu')(tower_12)
    tower_12 = Conv2D(num_filters*2, (3, 3), padding='valid', activation='relu')(tower_12)

    # Tower 3
    tower_3 = Conv2D(num_filters, (1, 1), padding='valid', activation='relu')(input_img)
    tower_3 = Conv2D(num_filters, (7, 7), padding='valid', activation='relu')(tower_3)

    # Merge Tower 12 and Tower 3
    tower_123 = concatenate([tower_12, tower_3], axis=-1)
    tower_123 = Conv2D(num_filters*3, (1, 1), padding='valid', activation='relu')(tower_123)
    tower_123 = Conv2D(num_filters*3, (2, 2), padding='valid', activation='relu')(tower_123)

    # Tower 4
    tower_4 = Conv2D(num_filters, (1, 1), padding='valid', activation='relu')(input_img)
    tower_4 = Conv2D(num_filters, (8, 8), padding='valid', activation='relu')(tower_4)

    # Merge Tower 123 and Tower 4
    tower_1234 = concatenate([tower_123, tower_4], axis=-1)
    tower_1234 = Conv2D(num_filters*4, (1, 1), padding='valid', activation='relu')(tower_1234)

    x = Flatten()(tower_1234)
    output = Dense(embedding_size, activation='relu')(x)

    return Model(inputs=input_img, outputs=output, name='embedder')

# Current net does not use bn. Retrain and then change this param to true
# Ep41 - loss: 0.1330 - acc: 0.9439 - val_loss: 0.1417 - val_acc: 0.9400
def conv_tower(num_filters=128, embedding_size=128, include_rights=True, bn=True):
    input_img = Input(shape=(8, 8, 17)) if include_rights else Input(shape=(8, 8, 13))

    # Block 0 - Square View
    x = Conv2D(num_filters, (1, 1), padding='same')(input_img)
    x = BatchNormalization()(x) if bn else x
    x = Activation('relu')(x)

    # Block 1 - Global
    x = ZeroPadding2D(padding=2)(x)
    x = Conv2D(num_filters, (8, 8), padding='valid')(x)
    x = BatchNormalization()(x) if bn else x
    x = Activation('relu')(x)

    # Block 2 - Large View
    x = Conv2D(num_filters, (5, 5), padding='same')(x)
    x = BatchNormalization()(x) if bn else x
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (5, 5), padding='same')(x)
    x = BatchNormalization()(x) if bn else x
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=1)(x)
    x = Conv2D(num_filters, (5, 5), padding='valid')(x)
    x = BatchNormalization()(x) if bn else x
    x = Activation('relu')(x)

    # Block 3 - Local View
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x) if bn else x
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x) if bn else x
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x) if bn else x
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x) if bn else x
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x) if bn else x
    x = Activation('relu')(x)

    x = Flatten()(x)
    output = Dense(embedding_size, activation='relu')(x)

    return Model(inputs=input_img, outputs=output, name='embedder')

def distilled_conv_tower(num_filters=128, embedding_size=128, include_rights=True, bn=True):

    input_img = Input(shape=(8, 8, 17)) if include_rights else Input(shape=(8, 8, 13))

    x = ZeroPadding2D(padding=2)(input_img)
    x = Conv2D(num_filters, (8, 8), padding='valid')(x)
    x = BatchNormalization()(x) if bn else x
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=1)(x)
    x = Conv2D(num_filters, (5, 5), padding='valid')(x)
    x = BatchNormalization()(x) if bn else x
    x = Activation('relu')(x)

    x = Conv2D(num_filters, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x) if bn else x
    x = Activation('relu')(x)

    x = Flatten()(x)
    output = Dense(embedding_size, activation='relu')(x)

    return Model(inputs=input_img, outputs=output, name='embedder')

# Comparators

def full_comparator(tower_func, optimizer=None, include_rights=True, embed=False):

    b1 = Input(shape=(8, 8, 17)) if include_rights else Input(shape=(8, 8, 13))
    b2 = Input(shape=(8, 8, 17)) if include_rights else Input(shape=(8, 8, 13))

    tower = tower_func(include_rights=include_rights)

    w = tower(b1)
    x = tower(b2)

    merged_vector = concatenate([w, x], axis=-1)
    output = Dense(1, activation='sigmoid', name='comparer')(merged_vector)

    if embed:
        outputs = [w, x, output]
    else:
        outputs = output

    model = Model(inputs=[b1, b2], outputs=outputs)

    loss = {'comparer': 'binary_crossentropy'}
    if embed:
        loss['embedder'] = 'mean_squared_error'

    model.compile(optimizer if optimizer is not None else 'adam',
                  loss, metrics={'comparer': 'accuracy'})

    return model

def split_comparator(tower_func, include_rights=True):

    b = Input(shape=(8, 8, 17)) if include_rights else Input(shape=(8, 8, 13))
    tower = tower_func(include_rights=include_rights)
    vec = tower(b)

    embedder = Model(inputs=b, outputs=vec)

    vec1, vec2 = Input(shape=tower.output_shape[1:]), Input(shape=tower.output_shape[1:])
    merged_vector = concatenate([vec1, vec2], axis=-1)
    output = Dense(1, activation='sigmoid', name='comparer')(merged_vector)

    comparer = Model(inputs=[vec1, vec2], outputs=output)

    return embedder, comparer

def comparator(tower_func, optimizer=None, include_rights=True, mode='train'):
    if mode == 'play':
        return split_comparator(tower_func, include_rights)
    elif mode == 'distill':
        return full_comparator(tower_func, optimizer, include_rights, embed=True)
    else:
        return full_comparator(tower_func, optimizer, include_rights)

def conv_comparator(optimizer=None, include_rights=True, mode='train'):

    return comparator(conv_tower, optimizer, include_rights, mode)

def distilled_conv_comparator(optimizer=None, include_rights=True, mode='train'):

    return comparator(distilled_conv_tower, optimizer, include_rights, mode)


### DEPRECATED ###
# To Un-Deprecate, increase the last dim of the input by 4

def modified_inception_module(input_img, num_filters):

    tower_1 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_img)
    tower_1 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(num_filters, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_img)
    # tower_1 = Conv2D(num_filters, 3, 3, padding='same', activation='relu')(tower_3)

    output = concatenate([tower_1, tower_2, tower_3], axis=-1)
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


# Ep43
# loss: 0.1846 - acc: 0.9205
# val_loss: 0.1824 - val_acc: 0.9217
def conv10layer(optimizer=None, play=False):

    return comparator(conv10layer_tower, play=play, optimizer=optimizer)

#Ep 28
def incpetion_net(optimizer=None, play=False):

    return comparator(inception_tower, play=play, optimizer=optimizer)

def distilled_inception_net(optimizer=None, play=False):

    distilled_tower = lambda: inception_tower(filters=[128, 128])
    return comparator(distilled_tower, play=play, optimizer=optimizer)

def conv_searcher(optimizer=None, play=False):

    return comparator(conv_searcher_tower, play=play, optimizer=optimizer)
