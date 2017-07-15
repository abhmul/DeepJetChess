import sys
import h5py
import numpy as np
from tqdm import tqdm

from np_utils import frontpad, backpad

def get_start_inds(M_h5):
    return frontpad(np.where(np.array(M_h5, dtype=np.int8)==0)[0][:-1] + 1, 0)

def ravel_game_ind(game_ind, start_inds_list):
    for i, start_inds in enumerate(start_inds_list):
        if game_ind >= len(start_inds):
            game_ind -= len(start_inds)
        else:
            return i, game_ind

def get_or_value(arr, ind, val):
    if ind == len(arr):
        return val
    else:
        return arr[ind]

def merge_h5(fname_out, *args, shuffle=False):

    board_sizes = []
    datasets = []
    start_inds_list = []

    for arg in args:
        print("Opening file %s" % arg)
        h5f = h5py.File(arg, 'r')
        # Read in the h5 file and validate it
        datasets.append([h5f[group] for group in ('X', 'M', 'W', 'C', 'E')])
        X, M, W, C, E = datasets[-1]
        assert(X.shape[0] == M.shape[0] == W.shape[0] == C.shape[0] == E.shape[0])
        # Get the number of boards in the file
        board_sizes.append(X.shape[0])
        start_inds_list.append(get_start_inds(M))

    moves = sum(board_sizes)
    num_games = sum(len(start_inds) for start_inds in start_inds_list)
    print("%s Total Boards" % moves)
    print("%s Total Games" % num_games)

    unraveled_game_inds = np.arange(num_games)
    if shuffle:
        np.shuffle(unraveled_game_inds)
    game_inds = np.asarray([ravel_game_ind(game_ind, start_inds_list) for game_ind in unraveled_game_inds])

    # Initialize the new h5 file
    g = h5py.File(fname_out, 'w')
    X_merged = g.create_dataset('X', (moves, 8, 8), dtype='i8', maxshape=(None, 8, 8), chunks=True)
    M_merged = g.create_dataset('M', (moves,), dtype='i8', maxshape=(None,), chunks=True)
    W_merged = g.create_dataset('W', (moves,), dtype='i8', maxshape=(None,), chunks=True)
    C_merged = g.create_dataset('C', (moves, 4), dtype='i8', maxshape=(None, 4), chunks=True)
    E_merged = g.create_dataset('E', (moves, 2), dtype='i8', maxshape=(None, 2), chunks=True)

    # Fill it in
    offset = 0
    for i, (set_ind, game_ind) in tqdm(enumerate(game_inds)):
        start_ind = start_inds_list[set_ind][game_ind]
        next_start_ind = get_or_value(start_inds_list[set_ind], game_ind + 1, board_sizes[set_ind])
        game_len = next_start_ind - start_ind
        # Discard any games that are really short
        if game_len < 4:
            continue
        X_merged[offset:offset + game_len] = datasets[set_ind]['X'][start_ind:next_start_ind]
        M_merged[offset:offset + game_len] = datasets[set_ind]['M'][start_ind:next_start_ind]
        W_merged[offset:offset + game_len] = datasets[set_ind]['W'][start_ind:next_start_ind]
        C_merged[offset:offset + game_len] = datasets[set_ind]['C'][start_ind:next_start_ind]
        E_merged[offset:offset + game_len] = datasets[set_ind]['E'][start_ind:next_start_ind]
        offset += game_len

    if offset < moves:
        [d.resize(size=offset, axis=0) for d in (X_merged, M_merged, W_merged, C_merged, E_merged)]

    # Flush and close
    g.flush()
    g.close()
    print("Dataset Merged into %s!" % fname_out)

if __name__ == "__main__":
    merge_h5(sys.argv[1], *sys.argv[2:])
