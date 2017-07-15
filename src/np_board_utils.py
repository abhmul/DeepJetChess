from functools import total_ordering

import numpy as np
import chess

TRANSLATE_PIECES = str.maketrans(".pnbrqkPNBRQK", "\x06" + "\x05\x04\x03\x02\x01\x00" + "\x07\x08\x09\x0A\x0B\x0C")
TRANSLATE_VALUES = str.maketrans("\n6543210789ABC", "/.pnbrqkPNBRQK")
CASTLE_FEN_ARRAY = np.array(["K", "Q", "k", "q"])

def make_gn_child(gn_current, move):
    gn_child = chess.pgn.GameNode()
    gn_child.parent = gn_current
    gn_child.move = move
    return gn_child

def switch_board_sides(X_game):
    """Switches sides for a numpy array of many boards"""
    # Flip the boards and invert the pieces
    return -1 * X_game[:, ::-1, ...]

def switch_castling_right_sides(castling_rights):
    return castling_rights[:, np.array([2, 3, 0, 1])]

def switch_to_moves(to_moves):
    return 1 - to_moves

def switch_state_sides(boards, castling_rights, en_passants, to_moves=None):
    new_boards = switch_board_sides(boards)
    new_castling_rights = switch_castling_right_sides(castling_rights)
    new_en_passants = flip_inds(en_passants)
    if to_moves is None:
        return new_boards, new_castling_rights, new_en_passants
    new_to_moves = switch_to_moves(to_moves)
    return new_boards, new_castling_rights, new_en_passants, new_to_moves

def switch_input_sides(model_input):
    new_model_input = np.empty(model_input.shape, dtype=model_input.dtype)
    # Switch the boards
    new_model_input[:, :, :, :6] = model_input[:, ::-1, :, 6:12]
    new_model_input[:, :, :, 6:12] = model_input[:, ::-1, :, :6]
    # Switch the castling rights
    new_model_input[:, :, :, 12:16] = model_input[:, :, :, np.array([14, 15, 12, 13])]
    # Switch the en passant
    new_model_input[:, :, :, 16] = model_input[:, ::-1, :, 16]
    return new_model_input

def sb2array(board):
    """
    Turns a string representation of a board into an
    8 x 8 numpy array representation of the board.
    """
    return np.fromstring(board.replace(' ', '').replace('\n', '').translate(TRANSLATE_PIECES), dtype=np.int8).reshape((8, 8)) - 6

def array2boardfen(arr):
    s = str(arr + 6).replace("[", "").replace("]", "").replace(" ", "").replace("10", "A").replace("11", "B").replace("12", "C").translate(TRANSLATE_VALUES)
    index_counts = []
    start = 0
    while True:
        count = 0
        ind = s.find('.', start)
        i = ind
        if i == -1: break
        count += 1
        i += 1
        while i != len(s) and s[i] == '.':
            count += 1
            i += 1
        index_counts.append((ind, count))
        start = i
    index_counts.sort(key=lambda tup: tup[1])
    prev_count = -1
    for ind, count in index_counts:
        if count != prev_count:
            s = s.replace(s[ind:ind+count], str(count))
        prev_count = count
    return s

def array2b(arr, turn, castling_rights, ep_square):
    b = chess.Board.empty()
    [b.set_piece_at(ind2square(i, j), chess.Piece(abs(arr[i, j]), arr[i, j] > 0)) for i in range(arr.shape[0]) for j in range(arr.shape[1]) if arr[i,j] != 0]
    # Set the turn
    b.turn = chess.WHITE if turn else chess.BLACK
    # Set the castling rights
    castle = "".join(CASTLE_FEN_ARRAY[castling_rights.astype(np.bool)])
    b.set_castling_fen(castle if castle != '' else '-')
    # Set the en passant squares
    if (ep_square[0], ep_square[1]) != (0, 0):
        b.ep_square = ind2square(ep_square[0], ep_square[1])
    return b

def square2ind(square):
    i = 7 - int(square / 8)
    j = square - (7 - i) * 8
    return i, j

def ind2square(i, j):
    return (7 - i) * 8 + j

def flip_inds(inds):
    return np.array([[1, -1]]) * (np.array([[7, 0]]) - inds)

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

    def switch_sides(self):
        self.turn = not self.turn
        self.np_board = switch_sides(self.np_board[np.newaxis])[0]

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
