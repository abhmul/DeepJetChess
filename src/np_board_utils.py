from functools import total_ordering

import numpy as np
import chess

TRANSLATE_PIECES = str.maketrans(".pnbrqkPNBRQK", "\x06" + "\x05\x04\x03\x02\x01\x00" + "\x07\x08\x09\x0A\x0B\x0C")

def switch_sides(X_game):
    """Switches sides for a numpy array of many boards"""
    # Flip the boards and invert the pieces
    return -1 * X_game[:, ::-1, ...]

def sb2array(board):
    """
    Turns a string representation of a board into an
    8 x 8 numpy array representation of the board.
    """
    return np.fromstring(board.replace(' ', '').replace('\n', '').translate(TRANSLATE_PIECES), dtype=np.int8).reshape((8, 8)) - 6

def sf2array(pos, flip):
    # Create a numpy array from a sunfish representation
    return np.fromstring(board.replace(' ', '').translate(TRANSLATE_PIECES), dtype=np.int8).reshape((8, 8)) - 6

def split_boards(X_board):
    """
    Turns the boards into 12 new boards for each piece on each side
    First 6 are the player's pieces and last 6 are the enemy pieces
    """
    X = np.empty(X_board.shape[:3] + (12,))
    # Do the player pieces first
    for pieceval in range(1, 7):
        # Select the squares with the specified piece
        X[:, :, :, pieceval-1] = (X_board == pieceval).astype(np.float32)
    # Next do the enemy pieces
    for pieceval in range(-1, -7, -1):
        # Select the squares with the specified piece
        X[:, :, :, -1*pieceval + 5] = (X_board == pieceval).astype(np.float32)
    return X

def create_input(arr_b, white_move):

    if arr_b.shape == (8, 8):
        n_samples = 1
        X = arr_b[np.newaxis]
        moves = np.asarray([white_move]).astype(np.float32)
        assert(moves.shape == (1,))
    elif arr_b.shape[1:] == (8, 8):
        n_samples = arr_b.shape[0]
        X = arr_b
        moves = np.asarray(white_moves).astype(np.float32)
        assert(moves.shape[0] == n_samples)
    else:
        raise ValueError("Invalid array shape: {}. Must be (8, 8) or (None, 8, 8)".format(arr_b.shape))

    arr_in = np.empty((n_samples, 8, 8, 13), dtype=np.float32)
    arr_in[:, :, :, :12] = split_boards(X)
    arr_in[:, :, :, 12] = moves[:, np.newaxis, np.newaxis]

    return arr_in

@total_ordering
class NpBoard(object):

    def __init__(self, board, comparator):
        self.__hash = board.zobrist_hash()
        self.turn = board.turn
        self.np_board = sb2array(str(board))
        self.comparator = comparator

    def __hash__(self):
        return self.__hash

    def __lt__(self, other):
        return self.comparator.less(self, other)

    def __eq__(self, other):
        return self.comparator.equal(self, other)

    def create_input(self):
        return create_input(self.np_board, self.turn)

    def is_win(self):
        return False

    def is_loss(self):
        return False


@total_ordering
class Win(NpBoard):

    def __init__(self):
        self.turn = None
        self.np_board = None

    def __hash__(self):
        raise ValueError("Cannot hash a Win board")

    def __lt__(self, other):
        return False

    def __eq__(self, other):
        return other.is_win()

    def is_win(self):
        return True

@total_ordering
class Loss(NpBoard):

    def __init__(self):
        self.turn = None
        self.np_board = None

    def __hash__(self):
        raise ValueError("Cannot hash a Loss board")

    def __lt__(self, other):
        return True

    def __eq__(self, other):
        return other.is_loss()

    def is_loss(self):
        return True

WIN = Win()
LOSS = Loss()
