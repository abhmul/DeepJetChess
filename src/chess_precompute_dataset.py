from collections import namedtuple
import numpy as np
import chess
import chess.pgn as pgn
import h5py
import pyjet.data as pyjet

from np_board_utils import split_boards

# Add Verbosity tuple to pyjet
VERBOSITY = namedtuple('VERBOSITY', ['QUIET', 'NORMAL', 'VERBOSE', 'DEBUG'])(0,1,2,3)

class ChessDataset(pyjet.Dataset):

    def __init__(self, fn_in, verbosity=VERBOSITY.NORMAL):
        super(ChessDataset, self).__init__()
        self.verbosity = verbosity

        # Initialize file stuff
        self.fn_in = fn_in

        # Load the dataset
        h5f = h5py.File(self.fn_in, 'r')
        self.log("Reading in file %s..." % self.fn_in, VERBOSITY.NORMAL)
        # Load the data
        # X, M, MM, W, C, E, S_ind, G_ind = [h5f[group] for group in ('X', 'M', 'MM', 'W', 'C', 'E', 'S_ind', 'G_ind')]
        X, MM, W, C, E, S_ind, G_ind = [h5f[group] for group in ('X', 'MM', 'W', 'C', 'E', 'S_ind', 'G_ind')]

        # Figure out which ones to load into memory and which to read off disk
        self.boards = X
        self.castling_rights = C
        self.en_passant = E
        # self.to_move = M
        self.results = W

        self.move_made_indicies = np.array(MM, dtype=np.uint8)

        self.state_inds = np.array(S_ind, dtype=int)
        self.game_inds = np.array(G_ind, dtype=int)

        self.range_num_states = np.arange(len(self.state_inds), dtype=int)

    def __len__(self):
        return len(self.game_inds)

    def get_game_slice(self, i):
        if i == len(self) - 1:
            return slice(self.game_inds[i], None)
        else:
            return slice(self.game_inds[i], self.game_inds[i+1])

    def get_state_slice(self, i):
        if i == len(self.state_inds) - 1:
            return slice(self.state_inds[i], None)
        else:
            return slice(self.state_inds[i], self.state_inds[i+1])

    def sample_state_ind(self, i):
        return np.random.choice(self.range_num_states[self.get_game_slice(i)])

    def create_batch(self, batch_game_inds, flatten=True):
        batch_state_range_inds = [self.sample_state_ind(ind) for ind in batch_game_inds]
        batch_board_slices = [self.get_state_slice(state_ind) for state_ind in batch_state_range_inds]

        samples = []
        labels = self.move_made_indicies[batch_state_range_inds]
        for board_slice in batch_board_slices:
            # Retrieve the data from the h5 datasets
            np_legal_boards = np.array(self.boards[board_slice], dtype=np.int8)
            castling_rights = np.array(self.castling_rights[board_slice], dtype=np.int8)
            en_passant_inds = np.array(self.en_passant[board_slice], dtype=np.int8) # num_legals, 2
            num_legals = len(np_legal_boards)
            # Create the sample
            sample = np.zeros((num_legals, 8, 8, 17))
            sample[:, :, :, :12] = split_boards(np_legal_boards)
            sample[:, :, :, 12:16] = castling_rights[:, np.newaxis, np.newaxis, :]
            # Figure out which boards have en passants
            have_eps = np.where(np.any(en_passant_inds != 0, axis=1))[0]
            sample[have_eps, en_passant_inds[have_eps, 0], en_passant_inds[have_eps, 1], 16]  = 1
            # Transpose for pytorch format
            sample = sample.transpose(0, 3, 1, 2)
            if flatten:
                flat_sample = np.zeros((num_legals, 8 * 8 * 13 + 4))
                flat_sample[:, :8 * 8 * 12] = sample[:, :12].reshape(num_legals, -1)
                flat_sample[:, 8*8*12:8*8*13] = sample[:, 16:].reshape(num_legals, -1)
                flat_sample[:, -4:] = castling_rights
                sample = flat_sample
            # Add the sample to the sample list
            samples.append(sample)
        return samples, labels.astype(int)

    def train_val_split(self, split=0.2, permute=True, seed=None):
        if seed is not None:
            np.random.seed(seed)
        full_index_array = np.random.permutation(len(self)) if permute else np.arange(len(self)).astype(int)
        split_ind = int(0.2 * len(self))
        train_index_array = full_index_array[split_ind:]
        val_index_array = full_index_array[:split_ind]
        return SplitChessDataset(self, train_index_array, verbosity=self.verbosity), SplitChessDataset(self, val_index_array, verbosity=self.verbosity)


class SplitChessDataset(pyjet.Dataset):

    def __init__(self, full_dataset, index_array, verbosity=VERBOSITY.NORMAL):
        super(SplitChessDataset, self).__init__()
        self.verbosity = verbosity

        self.full_dataset = full_dataset
        self.index_array = index_array

    def __len__(self):
        return len(self.index_array)

    def create_batch(self, batch_inds):
        batch_game_inds = self.index_array[batch_inds]
        return self.full_dataset.create_batch(batch_game_inds)
