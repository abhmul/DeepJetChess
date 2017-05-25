from __future__ import print_function

import sys
import os
from collections import namedtuple

import numpy as np
import h5py
import chess
from keras.utils import to_categorical

from np_board_utils import switch_sides, split_boards

# ================ Meta ====================
__description__ = 'Chess HDF5 Processing Engine'
__version__ = '0.1.0'
__license__ = 'MIT'
__author__ = 'Abhijeet Mulgund (@abhmul)'
__status__ = "Prototype"
__copyright__ = "Copyright 2017"

def package_versions():
    print('chessengine   \t', __version__)
    print('python        \t', sys.version[0:5])
    print('numpy         \t', np.__version__)
    print('h5py          \t', h5py.__version__)
    print('chess         \t', chess.__version__)

SOURCEDIR = os.getcwd()

VERBOSITY = namedtuple('VERBOSITY', ['QUIET', 'NORMAL', 'VERBOSE', 'DEBUG'])(0,1,2,3)

class DataProcessor(object):

    def __init__(self, exclude_n=10, exclude_capture=True, include_rights=True, verbosity=VERBOSITY.NORMAL):
        self.exclude_n = exclude_n
        self.exclude_capture = exclude_capture
        self.include_rights = include_rights
        self.verbosity = verbosity

        # Initialize file stuff
        self.fn_ins = []
        self.X_win = np.zeros((0, 8, 8), dtype=np.int8)
        self.M_win = np.zeros((0,), dtype=np.int8)
        self.C_win = np.zeros((0,), dtype=np.int8)
        self.X_loss = np.zeros((0, 8, 8), dtype=np.int8)
        self.M_loss = np.zeros((0,), dtype=np.int8)
        self.C_loss = np.zeros((0,), dtype=np.int8)

        self.win_boards = 0
        self.loss_boards = 0

    def log(self, statement, verbosity):
        if self.verbosity >= verbosity:
            print(statement)

    def filter_boards(self, X_arr, end_inds):

        mask = np.ones(len(X_arr), dtype=np.bool)
        if self.exclude_n:
            for i in range(self.exclude_n):
                mask[end_inds[:-1] + i] = 0

            excluded = len(mask) - np.count_nonzero(mask)
            self.log("Excluded Start Moves: %s" % excluded, VERBOSITY.VERBOSE)
        if self.exclude_capture:
            pieces = np.count_nonzero(X_arr, axis=(1, 2))
            mask[np.where(pieces == (np.roll(pieces, 1) - 1))] = 0

            excluded = len(mask) - np.count_nonzero(mask) - excluded
            self.log("Excluded Capture Moves: %s" % excluded, VERBOSITY.VERBOSE)

        return mask

    def load_many_h5(self, fn_ins):

        for fn_in in fn_ins:
            self.load_h5(fn_in)

    def load_h5(self, fn_in):

        h5f = h5py.File(fn_in, 'r')
        self.log("Reading in file %s..." % fn_in, VERBOSITY.NORMAL)
        # Load the data
        if self.include_rights:
            X, M, W, C = [np.array(h5f[group], dtype=np.int8) for group in ('X', 'M', 'W', 'C')]
        else:
            X, M, W = [np.array(h5f[group], dtype=np.int8) for group in ('X', 'M', 'W')]

        # Make sure that we have the same number of boards as we do moves.
        assert(X.shape[0] == M.shape[0])
        num_moves = X.shape[0]
        self.log('%s moves in dataset' % num_moves, VERBOSITY.NORMAL)

        # Find the end game turns
        # BUG: Have to add 1 so end index is right... not sure why??
        end_inds = np.where(M == 0)[0] + 1

        # Make sure that we have the same number of ends to games as we do winners
        # assert(len(end_inds))
        num_games = len(end_inds)
        self.log('%s games in dataset' % num_games, VERBOSITY.NORMAL)

        # Filter out all the specified moves
        mask = self.filter_boards(X, end_inds)
        X = X[mask]
        M = M[mask]
        W = W[mask]
        if self.include_rights:
            C = C[mask]

        # Seperate the games where white wins and loses
        win_mask = np.where(W == 1)
        loss_mask = np.where(W == -1)
        W = None

        X_win, X_loss = X[win_mask], X[loss_mask]
        X = None
        M_win, M_loss = M[win_mask], M[loss_mask]
        M = None
        if self.include_rights:
            C_win, C_loss = C[win_mask], C[loss_mask]
            C = None

        self.X_win = np.concatenate([self.X_win, X_win], axis=0)
        self.X_loss = np.concatenate([self.X_loss, X_loss], axis=0)
        self.M_win = np.concatenate([self.M_win, M_win], axis=0)
        self.M_loss = np.concatenate([self.M_loss, M_loss], axis=0)
        if self.include_rights:
            self.C_win = np.concatenate([self.C_win, C_win], axis=0)
            self.C_loss = np.concatenate([self.C_loss, C_loss], axis=0)

        self.win_boards = len(self.M_win)
        self.loss_boards = len(self.M_loss)

        self.log("%s win boards" % self.win_boards, VERBOSITY.NORMAL)
        self.log("%s loss boards" % self.loss_boards, VERBOSITY.NORMAL)

        self.fn_ins.append(fn_in)

    @staticmethod
    def _unwind_castling_rights(right_flag):
        wk = right_flag & 1
        wq = right_flag & 2
        bk = right_flag & 4
        bq = right_flag & 8
        return wk, wq, bk, bq

    @staticmethod
    def _vec_unwind_castling_rights(right_flags):
        return np.asarray([np.asarray(DataProcessor._unwind_castling_rights(right_flag)) for right_flag in right_flags])

    def _create_batch(self, tot_win_inds, tot_loss_inds, batch_size=32, flat=False):

        # Get the win indicies and loss indicies
        self.log("\tPicking %s random win inds from range %s to %s" % (batch_size, tot_win_inds[0], tot_win_inds[-1]), VERBOSITY.DEBUG)
        win_inds = np.random.choice(tot_win_inds, batch_size)
        self.log(win_inds, VERBOSITY.DEBUG)
        self.log("\tPicking %s random loss inds from range %s to %s" % (batch_size, tot_loss_inds[0], tot_loss_inds[-1]), VERBOSITY.DEBUG)
        loss_inds = np.random.choice(tot_loss_inds, batch_size)
        self.log(loss_inds, VERBOSITY.DEBUG)

        # Get the labels (whether b1 is better)
        y = np.random.choice(2, batch_size)

        # Get the color for the board 1
        b1colors = np.random.choice(2, batch_size)

        # Initialize the batches
        if self.include_rights:
            x1_batch = np.empty((batch_size, 8, 8, 17))
            x2_batch = np.empty((batch_size, 8, 8, 17))
        else:
            x1_batch = np.empty((batch_size, 8, 8, 13))
            x2_batch = np.empty((batch_size, 8, 8, 13))

        x_win = self.X_win[win_inds]
        x_loss = self.X_loss[loss_inds]
        m_win = self.M_win[win_inds]
        m_loss = self.M_loss[loss_inds]
        if self.include_rights:
            c_win = self._vec_unwind_castling_rights(self.C_win[win_inds])
            c_loss = self._vec_unwind_castling_rights(self.C_loss[loss_inds])

            assert(c_win.shape == c_loss.shape == (batch_size, 4))



        # Get the masks for the 4 different possibilities
        white1_win = np.logical_and(y, b1colors)
        black1_win = np.logical_and(y, np.logical_not(b1colors))
        white1_loss = np.logical_and(np.logical_not(y), b1colors)
        black1_loss = np.logical_and(np.logical_not(y), np.logical_not(b1colors))

        assert(np.sum([white1_win, black1_win, white1_loss, black1_loss]) == batch_size)

        x1_batch[white1_win, :, :, 0:12] = split_boards(x_win[white1_win])
        x1_batch[white1_win, :, :, 12] = m_win[white1_win, np.newaxis, np.newaxis]
        x2_batch[white1_win, :, :, 0:12] = split_boards(x_loss[white1_win])
        x2_batch[white1_win, :, :, 12] = m_loss[white1_win, np.newaxis, np.newaxis]



        x1_batch[black1_win, :, :, 0:12] = split_boards(switch_sides(x_loss[black1_win]))
        x1_batch[black1_win, :, :, 12] = np.logical_not(m_loss[black1_win, np.newaxis, np.newaxis])
        x2_batch[black1_win, :, :, 0:12] = split_boards(switch_sides(x_win[black1_win]))
        x2_batch[black1_win, :, :, 12] = np.logical_not(m_win[black1_win, np.newaxis, np.newaxis])



        x1_batch[white1_loss, :, :, 0:12] = split_boards(x_loss[white1_loss])
        x1_batch[white1_loss, :, :, 12] = m_loss[white1_loss, np.newaxis, np.newaxis]
        x2_batch[white1_loss, :, :, 0:12] = split_boards(x_win[white1_loss])
        x2_batch[white1_loss, :, :, 12] = m_win[white1_loss, np.newaxis, np.newaxis]


        x1_batch[black1_loss, :, :, 0:12] = split_boards(switch_sides(x_win[black1_loss]))
        x1_batch[black1_loss, :, :, 12] = np.logical_not(m_win[black1_loss, np.newaxis, np.newaxis])
        x2_batch[black1_loss, :, :, 0:12] = split_boards(switch_sides(x_loss[black1_loss]))
        x2_batch[black1_loss, :, :, 12] = np.logical_not(m_loss[black1_loss, np.newaxis, np.newaxis])

        if self.include_rights:
            x1_batch[white1_win, :, :, 13:] = c_win[white1_win, np.newaxis, np.newaxis]
            x2_batch[white1_win, :, :, 13:] = c_loss[white1_win, np.newaxis, np.newaxis]
            x1_batch[black1_win, :, :, 13:] = c_loss[black1_win, np.newaxis, np.newaxis]
            x2_batch[black1_win, :, :, 13:] = c_win[black1_win, np.newaxis, np.newaxis]
            x1_batch[white1_loss, :, :, 13:] = c_loss[white1_loss, np.newaxis, np.newaxis]
            x2_batch[white1_loss, :, :, 13:] = c_win[white1_loss, np.newaxis, np.newaxis]
            x1_batch[black1_loss, :, :, 13:] = c_win[black1_loss, np.newaxis, np.newaxis]
            x2_batch[black1_loss, :, :, 13:] = c_loss[black1_loss, np.newaxis, np.newaxis]

        if flat:
            new_x1_batch = np.empty((batch_size, 773))
            new_x1_batch[:, :768] = x1_batch[:, :, :, 0:12].transpose(0, 3, 1, 2).reshape(batch_size, 768)
            new_x1_batch[:, 768] = x1_batch[:, 0, 0, 12]
            new_x1_batch[:, 769:] = x1_batch[:, 0, 0, 13:]
            new_x2_batch = np.empty((batch_size, 773))
            new_x2_batch[:, :768] = x2_batch[:, :, :, 0:12].transpose(0, 3, 1, 2).reshape(batch_size, 768)
            new_x2_batch[:, 768] = x2_batch[:, 0, 0, 12]
            new_x2_batch[:, 769:] = x2_batch[:, 0, 0, 13:]
            x1_batch = new_x1_batch
            x2_batch = new_x2_batch
            y = to_categorical(np.logical_not(y), num_classes=2)

        return [x1_batch.astype(np.float32), x2_batch.astype(np.float32)], y.astype(np.float32)

    def chessgen(self, tot_win_inds, tot_loss_inds, batch_size=32, flat=False):

        while True:
            yield self._create_batch(tot_win_inds, tot_loss_inds, batch_size, flat)

    def distillgen(self, net, mode, graph=None):

        gen = self.train_gen if mode == 'train' else self.val_gen
        assert(graph is not None)
        with graph.as_default():
            while True:
                [x1, x2], y = next(gen)
                e1, e2, pred = net.predict_on_batch([x1, x2])
                yield [x1, x2], [e1, e2, pred]

    def create_gen(self, batch_size=32, test_split=0.15, random_state=None, flat=False):
        if random_state is not None:
            np.random.seed(random_state)

        win_split_ind = int((1 - test_split) * self.win_boards)
        loss_split_ind = int((1 - test_split) * self.loss_boards)

        all_win_inds = np.arange(self.win_boards)
        all_loss_inds = np.arange(self.loss_boards)

        np.random.shuffle(all_win_inds)
        np.random.shuffle(all_loss_inds)

        train_win_inds = all_win_inds[:win_split_ind]
        train_loss_inds = all_loss_inds[:loss_split_ind]
        val_win_inds = all_win_inds[win_split_ind:]
        val_loss_inds = all_loss_inds[loss_split_ind:]

        self.train_gen = self.chessgen(train_win_inds, train_loss_inds, batch_size, flat)
        self.val_gen = self.chessgen(val_win_inds, val_loss_inds, batch_size, flat)
        return self.train_gen, self.val_gen
