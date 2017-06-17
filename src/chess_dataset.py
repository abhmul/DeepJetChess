from collections import namedtuple
import numpy as np
import chess
import chess.pgn as pgn
import h5py

from np_board_utils import sb2array, array2b, make_gn_child, split_boards, switch_state_sides, switch_board_sides, square2ind
from np_utils import frontpad, backpad

import pyjet.data as pyjet


# Add Verbosity tuple to pyjet
VERBOSITY = namedtuple('VERBOSITY', ['QUIET', 'NORMAL', 'VERBOSE', 'DEBUG'])(0,1,2,3)

class ChessDataset(pyjet.Dataset):
    """docstring for ."""
    def __init__(self, verbosity=VERBOSITY.NORMAL):
        super(ChessDataset, self).__init__()
        self.verbosity = verbosity

        # Initialize file stuff
        self.fn_ins = []
        self._game_ids = np.empty((0,), dtype=np.int8)
        self._boards = np.zeros((0, 8, 8), dtype=np.int8)
        self._castling_rights = np.zeros((0, 4), dtype=np.int8)
        self._en_passant = np.zeros((0, 2), dtype=np.int8)
        self._to_move = np.empty((0,), dtype=np.int8)
        self._results = np.empty((0,), dtype=np.int8)
        self._start_inds = None
        self._board_inds = None

    def __len__(self):
        return len(self.start_inds)

    def load_many_h5(self, fn_ins):

        for fn_in in fn_ins:
            self.load_h5(fn_in)

    def load_h5(self, fn_in):

        h5f = h5py.File(fn_in, 'r')
        self.log("Reading in file %s..." % fn_in, VERBOSITY.NORMAL)
        # Load the data
        X, M, W, C, E = [np.array(h5f[group], dtype=np.int8) for group in ('X', 'M', 'W', 'C', 'E')]

        while True:
            # Find the start game turns
            start_inds = frontpad(np.where(M==0)[0][:-1] + 1, 0)
            shifted_start_inds = backpad(start_inds[1:], len(X))
            # Number the boards based on game number
            game_ids = np.zeros((len(X),), dtype=np.int32)
            for game_i in range(len(start_inds)):
                game_ids[start_inds[game_i]:shifted_start_inds[game_i]] = game_i + self.num_game_ids

            # Subtract start inds from itself to find the short games
            mask = np.ones((len(X),), dtype=np.bool)
            start_diff = (shifted_start_inds - start_inds) <= 2
            # print(start_diff)
            # print(shifted_start_inds[-1])
            # print(start_inds[-1])
            # If none of the games are shorter than or equal to two boards, we're done
            if not np.any(start_diff):
                break
            self.log("Cleaning up the short games from the dataset", VERBOSITY.NORMAL)
            for i in range(len(start_diff)):
                if start_diff[i]:
                    mask[start_inds[i]:shifted_start_inds[i]] = False
            # Apply the mask
            X = X[mask]
            M = M[mask]
            W = W[mask]
            C = C[mask]
            E = E[mask]

        # Make sure that we have the same number of boards as we do moves.
        assert(X.shape[0] == M.shape[0])
        num_moves = X.shape[0]
        self.log('%s moves in dataset' % num_moves, VERBOSITY.NORMAL)

        # Make sure that we have the same number of ends to games as we do winners
        # assert(len(start_inds))
        num_games = len(start_inds)
        self.log('%s games in dataset' % num_games, VERBOSITY.NORMAL)
        start_inds = None

        self._game_ids = np.concatenate([self._game_ids, game_ids], axis=0)
        game_ids = None
        self._start_inds = None
        self._boards = np.concatenate([self._boards, X], axis=0)
        X = None
        self._to_move = np.concatenate([self._to_move, M], axis=0)
        M = None
        self._results = np.concatenate([self._results, W == 1], axis=0)
        W = None
        self._castling_rights = np.concatenate([self._castling_rights, C], axis=0)
        C = None
        self._en_passant = np.concatenate([self._en_passant, E], axis=0)
        E = None

        assert(len(self._boards) == len(self._to_move) == len(self._results))

        self.log("%s boards" % len(self._boards), VERBOSITY.NORMAL)

        self.fn_ins.append(fn_in)

    @property
    def num_game_ids(self):
        if len(self._game_ids):
            return self._game_ids[-1] + 1
        return 0

    @property
    def board_inds(self):
        if self._board_inds is None:
            self._board_inds = np.arange(len(self._boards))
        return self._board_inds

    @property
    def start_inds(self):
        if self._start_inds is None:
            self._start_inds = np.unique(np.where((self.game_ids - frontpad(self.game_ids[:-1], -1)) != 0)[0])
        return self._start_inds

    @property
    def game_ids(self):
        return self._game_ids

    def get_game_slice(self, i, exclude_final=False):
        result = self._results[self.start_inds[i]]
        if i == len(self) - 1:
            return slice(self.start_inds[i] + (1-result), (-1 if exclude_final else None), 2)
        else:
            return slice(self.start_inds[i]  + (1-result), (self.start_inds[i+1] - (1 if exclude_final else 0)), 2)

    def sample_board(self, i):
        """Samples one board from game i"""
        # game_slice = self.get_game_slice(i, exclude_final=True)
        # valid_boards = self.board_inds[game_slice]
        # if len(valid_boards) == 0:
        #     print("Game: ", i)
        #     print("Slice: ", game_slice)
        #     print("Surrounding start inds: ", self.start_inds[i-2:i+2])
        #     print("Game Boards: ", self._boards[self.start_inds[i]-2:self.start_inds[i+1] + 2])
        # return np.random.choice(valid_boards)
        return np.random.choice(self.board_inds[self.get_game_slice(i, exclude_final=True)])

    def create_batch(self, batch_game_inds):
        """Randomly selects a board from each game in batch_game_inds"""
        batch_board_inds = np.array([self.sample_board(ind) for ind in batch_game_inds])
        batch_npboards = self._boards[batch_board_inds]
        batch_castling_rights = self._castling_rights[batch_board_inds]
        batch_en_passant = self._en_passant[batch_board_inds]
        batch_to_move = self._to_move[batch_board_inds]
        batch_results = self._results[batch_board_inds]
        # switch_state_sides works
        lost_mask = (batch_results == 0)
        batch_npboards[lost_mask], batch_castling_rights[lost_mask], batch_en_passant[lost_mask], batch_to_move[lost_mask] = \
            switch_state_sides(batch_npboards[lost_mask], batch_castling_rights[lost_mask], batch_en_passant[lost_mask], batch_to_move[lost_mask])
        x_batch = []
        y_batch = np.zeros((len(batch_game_inds,)), dtype=int)
        for j, board_ind in enumerate(batch_board_inds):
            board = array2b(batch_npboards[j], batch_to_move[j],
                            batch_castling_rights[j],
                            batch_en_passant[j])
            gn = pgn.Game.from_board(board)
            # Cache the board
            gn.board(_cache=True)
            gn_children = [make_gn_child(gn, move) for move in board.legal_moves]
            # Construct the numpy array to hold the legal moves
            np_legal_move_boards = np.empty((len(gn_children), 8, 8))
            batch_sample = np.zeros((len(gn_children), 8, 8, 17))
            for i, gn_child in enumerate(gn_children):
                child_board = gn_child.board()
                np_legal_move_boards[i] = sb2array(str(child_board))
                # Set castling rights
                if child_board.has_kingside_castling_rights(chess.WHITE):
                    batch_sample[i, :, :, 12] = 1
                if child_board.has_queenside_castling_rights(chess.WHITE):
                    batch_sample[i, :, :, 13] = 1
                if child_board.has_kingside_castling_rights(chess.BLACK):
                    batch_sample[i, :, :, 14] = 1
                if child_board.has_queenside_castling_rights(chess.BLACK):
                    batch_sample[i, :, :, 15] = 1
                # Set the en passant square
                if child_board.has_legal_en_passant():
                    row, col = square2ind(child_board.ep_square)
                    batch_sample[i, row, col, 16] = 1
            batch_sample[:, :, :, :12] = split_boards(np_legal_move_boards)

            # Figure out what the actual next move was
            next_npboard = self._boards[board_ind + 1]
            # Flip it if white lost
            if lost_mask[j] == 1:
                next_npboard = switch_board_sides(next_npboard[np.newaxis])
            categorical_label = np.all(np_legal_move_boards == next_npboard, axis=(1, 2))
            assert(categorical_label.shape == (len(gn_children),))
            # print("Current Board")
            # if lost_mask[j] == 1:
            #     print("Board was Flipped")
            # print(self._boards[board_ind])
            # print("Next npboard")
            # print(next_npboard)
            # print("Legal Boards:")
            # for i in range(len(np_legal_move_boards)):
            #     print(np_legal_move_boards[i])
            #     print(categorical_label[i])
            label = np.where(categorical_label)[0]
            assert(len(label) == 1)
            # input()
            label = label[0]
            # Change to pytorch format
            batch_sample = batch_sample.transpose(0, 3, 1, 2)
            x_batch.append(batch_sample)
            y_batch[j] = label
        return x_batch, y_batch
