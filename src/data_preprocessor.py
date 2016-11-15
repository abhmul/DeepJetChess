import h5py
import os
import numpy as np

def load_data():
    direc = '../'
    for fn in os.listdir(direc):
        if not fn.endswith('.hdf5'):
            continue

        fn = os.path.join(direc, fn)
        try:
            yield h5py.File(fn, 'r')
        except:
            print 'could not read', fn


def split_data(boards, moves_left, test_size=100000, val_split=.2):

    # Get all the indices that have a move following
    inds = np.arange(boards.shape[0])[np.where(moves_left != 0)]
    # Shuffle them
    np.random.shuffle(inds)
    # Get how big the training set will be
    train_size = int((len(inds) - test_size) * (1 - val_split))
    # Split the indices
    test_inds = inds[:test_size]
    train_inds = inds[test_size:test_size+train_size]
    val_inds = inds[test_size+train_size:]

    return train_inds, val_inds, test_inds


def get_data():

    datablocks = ('X', 'M')
    data = [[] for s in datablocks]
    for f in load_data():
        try:
            for i, s in enumerate(datablocks):
                data[i].append(f[s].value)
        except:
            raise('failed reading from %s' % f.filename)

    data = [np.array(s) for s in data]
    return data


def split_boards_seperate(boards, num_pieces=12):

    split_boards = np.zeros(boards.shape[:1] + (num_pieces,) + boards.shape[1:])
    for piece in xrange(-1 * num_pieces/2, num_pieces/2):
        if not piece:
            continue
        ind = piece + num_pieces/2 if piece < 0 else piece + num_pieces/2 -1
        split_boards[:, ind] = (boards == piece).astype(np.float32)

    return split_boards

def get_boards(boards, seperate=True):
    if seperate:
        return split_boards_seperate(boards)
    else:
        raise NotImplementedError

def get_labels(boards, inds):

    nxt_boards = np.fliplr(boards[inds+1])
    # Next boards will be from the opposite color so add instead of subtract
    # The multiplication comes from the fact that a piece will not be in the square another piece just moved from
    return ((boards + nxt_boards) * (nxt_boards != 0) != 0).astype(np.float32)


def train_batch_gen(boards, inds, batch_size=64, shuffle=True):
    # chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    # Get the number of batches
    number_of_batches = np.ceil(inds.shape[0] / batch_size)
    counter = 0
    sample_inds = inds
    # Shuffle the inds if we need to
    if shuffle:
        np.random.shuffle(sample_inds)
    # Do the batch generation
    while True:
        batch_index = sample_inds[batch_size * counter:batch_size * (counter + 1)]
        X_batch = get_boards(boards[batch_index], seperate=True)
        y_batch = get_labels(boards, batch_index)
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_inds)
            counter = 0
