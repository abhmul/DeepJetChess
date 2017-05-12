from __future__ import division, print_function
import numpy as np
import chess
import chess.pgn
import h5py
from multiprocessing import Pool
from timeit import timeit
import string
import re

import unittest as tst

from np_board_utils import sb2array

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

def sb2array2(board):
    """
    Turns a string board into an ndarray
    Black is represented as negative values
    White is represented as positive values
    Empty squares are 0
    """
    # Initialize the board
    x = np.zeros((8, 8), dtype=np.int8)
    # We start examining the board at position 0
    pos = [0, 0]
    # These are characters in the string not part of the actual board
    non_squares = {' ', '\n'}
    # These are our replacement values for each piece
    subs = {'.': 0, 'P': 1, 'R': 4, 'N': 2, 'B': 3, 'Q': 5, 'K': 6, 'p': -1, 'r': -4, 'n': -2,
            'b': -3, 'q': -5, 'k': -6}

    # Now loop over all the characters and fill in the matrix
    for piece in board:
        # If the piece is non-square, skip over it
        if piece in non_squares:
            continue
        # Otherwise if the square has a piece, update the 2darray
        elif piece is not '.':
            x[pos[0], pos[1]] = subs[piece]
        # Update the position to the next square
        pos[1] = (pos[1] + 1) % 8
        if not pos[1]:
            pos[0] += 1
    return x

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
    else:
        print('Game had invalid result %s, continuing to next game' % game_result)
        return None

    game_boards = build_game(game)
    # Create the move array and set the last move to 0
    move_arr = np.array([1 if board.turn else -1 for board in game_boards], dtype=np.int8)
    move_arr[-1] = 0

    # Initialize the board ndarray.
    x = np.empty((move_arr.shape[0], 8, 8), dtype=np.int8)
    # Fill in the board ndarray
    for i, board in enumerate(game_boards):
        x[i] = sb2array(str(board))

    return x, move_arr, winner

def to_h5(fname_in, fname_out):

    # Iniitialize the datasets
    g = h5py.File(fname_out, 'w')
    move_size = 8388607 # Empirically determined
    # game_size = 65535 # Empirically determined
    print("Initializing Dataset...")
    X = g.create_dataset('X', (move_size, 8, 8), dtype='i8', maxshape=(None, 8, 8), chunks=True)
    M = g.create_dataset('M', (move_size,), dtype='i8', maxshape=(None,), chunks=True)
    W = g.create_dataset('W', (move_size,), dtype='i8', maxshape=(None,), chunks=True)
    print("Dataset Initialized")

    moves = 0

    game_reader = read_games(fname_in)
    pool = Pool(processes=8)

    # Loop through every game
    for extract_try in pool.imap_unordered(extract_gdata, game_reader):
        # Try to extract the game data
        if extract_try is None:
            continue

        x, move_arr, winner = extract_try

        game_moves = len(move_arr)

        # Resize the h5 file if we're gonna need more space for boards and turns
        while moves + game_moves >= move_size:
            g.flush()
            move_size = 2 * move_size + 1
            print('Resizing board and moves to %s' % move_size)
            [d.resize(size=move_size, axis=0) for d in (X, M, W)]

        X[moves:moves+game_moves] = x
        M[moves:moves+game_moves] = move_arr
        W[moves:moves+game_moves] = winner

        moves += game_moves

    # Resize down to if we overshot
    print('Final Size for boards, moves, and winners: %s' % moves)
    [d.resize(size=moves, axis=0) for d in (X, M, W)]


if __name__ == '__main__':
    # tst.main()
    # a = timeit("sb2array(str(b))", setup="import chess;from parse_games import sb2array;b = chess.Board()", number=10000)
    # print("Original: %s" % a)
    # b = timeit("sb2array2(str(b))", setup="import chess;from parse_games import sb2array2;b = chess.Board()", number=10000)
    # print("New: %s" % b)
    to_h5(sys.argv[1], sys.argv[2])
