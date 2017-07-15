from __future__ import division, print_function
import numpy as np
import chess
import chess.pgn
import h5py
from multiprocessing import Pool
from timeit import timeit
import string
import re
from tqdm import tqdm

import unittest as tst

from np_board_utils import sb2array, square2ind

import sys
sys.setrecursionlimit(10000)

def read_games(fn):
    """
    A generator that generates every game in
    the .pgn file
    """
    f = open(fn)

    while True:
        try:
            g = chess.pgn.read_game(f)
        except KeyboardInterrupt:
            raise
        except:
            continue

        if not g:
            break

        yield g
    f.close()

def build_game(game):
    """Returns a list of all the boards in order in a game"""
    gn = game.end()
    game_boards = []
    # Loop through the game boards from end to start
    while gn:
        game_boards.append(gn.board())
        gn = gn.parent
    # Reverse so they're start to end
    game_boards.reverse()
    return game_boards

def extract_gdata(game):
    """
    Turns a game into an ndarray of board positions, a move array, and the winner
    The move array holds a value for each board position (-1 if it's black's move,
    1 if it's white's move, and 0 if the game is over)
    The winner is stored as -1 if black, 1 if white, and 0 if tie
    """
    # First we find the winner of the game
    results = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}
    game_result = game.headers['Result']
    # If we have a normal result,
    if game_result in results:
        winner = results[game_result]
        if not winner:
            # Throw out draws
            return None
    else:
        print('Game had invalid result %s, continuing to next game' % game_result)
        return None

    game_boards = build_game(game)
    # Create the move array and set the last move to 0
    move_arr = np.array([1 if board.turn else -1 for board in game_boards], dtype=np.int8)
    move_arr[-1] = 0

    # Initialize the board ndarray.
    x = np.empty((move_arr.shape[0], 8, 8), dtype=np.int8)
    c = np.zeros((move_arr.shape[0], 4), dtype=np.int8)
    ep = np.zeros((move_arr.shape[0], 2), dtype=np.int8)
    # Fill in the board ndarray
    for i, board in enumerate(game_boards):
        x[i] = sb2array(str(board))
        # Stores castling rights in the order WK WQ BK BQ
        if board.has_kingside_castling_rights(chess.WHITE):
            c[i, 0] = 1
        if board.has_queenside_castling_rights(chess.WHITE):
            c[i, 1] = 1
        if board.has_kingside_castling_rights(chess.BLACK):
            c[i, 2] = 1
        if board.has_queenside_castling_rights(chess.BLACK):
            c[i, 3] = 1
        # Figure out where en_passant can occur
        if board.has_legal_en_passant():
            ep[i, 0], ep[i, 1] = square2ind(board.ep_square)

    return x, move_arr, winner, c, ep

def resize_arr(arr, new_size, axis=0):
    extension_shape = list(arr.shape)
    extension_shape[axis] = new_size - extension_shape[axis]
    return np.concatenate([arr, np.zeros(extension_shape, dtype=arr.dtype)], axis=axis)

def to_h5(fname_in, fname_out):

    # Iniitialize the datasets

    move_size = 7000000 # Empirically determined
    # game_size = 65535 # Empirically determined
    print("Initializing Numpy Arrays")
    X_arr =np.zeros((move_size, 8, 8), dtype=np.int8)
    M_arr = np.zeros((move_size,), dtype=np.int8)
    W_arr = np.zeros((move_size,), dtype=np.int8)
    C_arr = np.zeros((move_size, 4), dtype=np.int8)
    E_arr = np.zeros((move_size, 2), dtype=np.int8)
    print("Numpy Arrays Initialized")

    moves = 0

    game_reader = read_games(fname_in)
    pool = Pool(processes=8)

    # Loop through every game
    for extract_try in pool.imap_unordered(extract_gdata, game_reader):
        # Try to extract the game data
        if extract_try is None:
            continue

        x, move_arr, winner, c, ep = extract_try

        game_moves = len(move_arr)

        # Resize the h5 file if we're gonna need more space for boards and turns
        while moves + game_moves >= move_size:
            move_size = int(1.1 * move_size) + 1
            print('Resizing board and moves to %s' % move_size)
            X_arr, M_arr, W_arr, C_arr, E_arr = [resize_arr(d, move_size) for d in (X_arr, M_arr, W_arr, C_arr, E_arr)]

        X_arr[moves:moves+game_moves] = x
        M_arr[moves:moves+game_moves] = move_arr
        W_arr[moves:moves+game_moves] = winner
        C_arr[moves:moves+game_moves] = c
        E_arr[moves:moves+game_moves] = ep


        moves += game_moves

    # Resize down to if we overshot
    print('Final Size for boards, moves, and winners: %s' % moves)
    print("Creating Dataset...")
    g = h5py.File(fname_out, 'w')
    X = g.create_dataset('X', (moves, 8, 8), data=X_arr[:moves], dtype='i8', maxshape=(None, 8, 8))
    M = g.create_dataset('M', (moves,), data=M_arr[:moves], dtype='i8', maxshape=(None,))
    W = g.create_dataset('W', (moves,), data=W_arr[:moves], dtype='i8', maxshape=(None,))
    C = g.create_dataset('C', (moves, 4), data=C_arr[:moves], dtype='i8', maxshape=(None, 4))
    E = g.create_dataset('E', (moves, 2), data=E_arr[:moves], dtype='i8', maxshape=(None, 2))
    g.flush()
    g.close()
    print("Dataset Created!")


if __name__ == '__main__':
    # tst.main()
    # a = timeit("sb2array(str(b))", setup="import chess;from parse_games import sb2array;b = chess.Board()", number=10000)
    # print("Original: %s" % a)
    # b = timeit("sb2array2(str(b))", setup="import chess;from parse_games import sb2array2;b = chess.Board()", number=10000)
    # print("New: %s" % b)
    to_h5(sys.argv[1], sys.argv[2])
