import chess, chess.pgn
import re
import unittest as tst
import numpy as np
import time
import sys
import os
import multiprocessing
import itertools
import random
import h5py


SUBS = {'\n': ' ', '.': '0', 'P': '1', 'R': '4', 'N': '2', 'B': '3', 'Q': '5', 'K': '6', 'p': '-1', 'r': '-4', 'n': '-2',
        'b': '-3', 'q': '-5', 'k': '-6'}

def read_games(fn):
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


def sb2array(board, flip=False):

    str_board = str(board)

    result = reduce(lambda sb, char: sb.replace(char, SUBS[char]), SUBS, str_board)
    x = np.fromstring(result, dtype=np.int8, sep=' ')
    x = x * -1 if flip else x
    return x


def bb2array2(b, turn):
    # Initialize the board
    x = np.zeros(64, dtype=np.int8)
    # We only flip the board if it is actually black's turn
    flip = not turn
    # We start examining the board at position 0
    pos = [0, 0]
    non_squares = {' ', '\n'}
    for piece in b:
        # If the piece is not a space, skip over it
        if piece in non_squares:
            continue
        elif piece is not '.':
            # Otherwise if the space is not empty add it to our numpy board
            row = pos[0]
            col = pos[1]
            color = 1
            if flip:
                row = 7 - pos[0]
                color = -1

            pieceval = color * int(SUBS[piece])

            x[row * 8 + col] = pieceval
        # Update our board position
        pos[1] = (pos[1] + 1) % 8
        pos[0] = pos[0] + 1 if not pos[1] else pos[0]

    return x

def parse_game(game):

    # Get the result of the game
    result_encoder = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}
    result = game.headers['Result']
    if result not in result_encoder:
        # We can't figure out who won or tied
        return None
    score = result_encoder[result]

    # Generate all the move boards
    gn = game.end()

    # Collect the moves
    game_moves = []
    moves_left = 0

    while gn:
        # We want to count all boards if draw, or just the winners board
        if not score or (gn.board().turn and score == 1) or (not gn.board().turn and score == -1):
            # Stores num moves left and the numpy array rep of the board
            game_moves.append((moves_left, bb2array2(str(gn.board()), gn.board().turn)))
        # Update the game node to its parent and the num moves left
        gn = gn.parent
        moves_left += 1

    # The game has no moves
    if not game_moves:
        return None

    # Remove the first board position
    game_moves.pop()

    return game_moves


def read_all_games(fname_in, fname_out):

    # Iniitialize the datasets
    g = h5py.File(fname_out, 'w')
    X = g.create_dataset('X', (0, 64), dtype='b', maxshape=(None, 64), chunks=True)
    M = g.create_dataset('M', (0,), dtype='b', maxshape=(None,), chunks=True)

    size = 0
    line = 0

    # Loop through every game
    for game in read_games(fname_in):
        game_moves = parse_game(game)
        if game_moves is None:
            continue
        # Loop through every board state in the game
        for moves_left, board in game_moves:
            x = board
            m = moves_left

            # Double the size if we're gonna run out
            if line + 1 >= size:
                g.flush()
                size = 2 * size + 1
                print 'Resizing to %s' % size
                [d.resize(size=size, axis=0) for d in (X, M)]

            # Add the board state to the data
            X[line] = x
            M[line] = m

            line += 1

    # Resize down to if we overshot
    print 'Final Size: %s' % line
    [d.resize(size=line, axis=0) for d in (X, M)]

    # Close the dataset
    g.close()


def time_test():
    fname = '../ficsgamesdb_201601_standard2000_nomovetimes_1416857.pgn'
    for i, game in enumerate(read_games(fname)):
        game1 = game.end()
        a = time.time()
        while game1:
            sb2array(game1.board(), flip=(not game1.board().turn))
            game1 = game1.parent
        print '1) sb2arr time: %s' % (time.time() - a)

        game2 = game.end()
        a = time.time()
        while game2:
            # print game2.board()
            # print game2.board().turn
            bb2array2(str(game2.board()), not game2.board().turn)
            game2 = game2.parent
        print '2) bb2arr time: %s' % (time.time() - a)

        if i == 10: break



class TestBoardMethods(tst.TestCase):

    def test_bb2array2(self):
        start_board = 'r n b q k b n r' \
                      'p p p p p p p p' \
                      '. . . . . . . .' \
                      '. . . . . . . .' \
                      '. . . . . . . .' \
                      '. . . . . . . .' \
                      'P P P P P P P P' \
                      'R N B Q K B N R'
        turn1 = True
        expected =np.array([[-4, -2, -3, -5, -6, -3, -2, -4],
                            [-1, -1, -1, -1, -1, -1, -1, -1,],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 1, 1, 1, 1, 1, 1, 1, 1],
                            [ 4, 2, 3, 5, 6, 3, 2, 4]])
        np.testing.assert_array_equal(expected, bb2array2(start_board, turn1).reshape(8,8))

        mid_board = 'r . b q . . k r' \
                    'p p p . . . p p' \
                    '. . n b . . . .' \
                    '. . . p p . . .' \
                    '. . . . . . . .' \
                    '. . N P . N . .' \
                    'P P P . Q P P P' \
                    'R . B . K . . R'
        turn2 = False
        expected = np.array([[-4, 0, -3, 0, -6, 0, 0, -4],
                             [-1, -1, -1, 0, -5, -1, -1, -1],
                             [ 0, 0, -2, -1, 0, -2, 0, 0],
                             [ 0, 0, 0, 0, 0, 0, 0, 0],
                             [ 0, 0, 0, 1, 1, 0, 0, 0],
                             [ 0, 0, 2, 3, 0, 0, 0, 0],
                             [ 1, 1, 1, 0, 0, 0, 1, 1],
                             [ 4, 0, 3, 5, 0, 0, 6, 4]])

        np.testing.assert_array_equal(expected, bb2array2(mid_board, turn2).reshape(8,8))
        np.testing.assert_array_equal(np.flipud(expected)*-1, bb2array2(mid_board, not turn2).reshape(8,8))

if __name__ == '__main__':
    fname = '../ficsgamesdb_201601_standard2000_nomovetimes_1416857.pgn'
    read_all_games(fname, '../test1.h5')
    # tst.main()