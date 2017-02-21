from __future__ import division, print_function
import numpy as np
import chess
import chess.pgn
import h5py

import unittest as tst

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

def b2array(board):
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
        x[i] = b2array(str(board))

    return x, move_arr, winner

def to_h5(fname_in, fname_out):

    # Iniitialize the datasets
    g = h5py.File(fname_out, 'w')
    X = g.create_dataset('X', (0, 8, 8), dtype='i8', maxshape=(None, 8, 8), chunks=True)
    M = g.create_dataset('M', (0,), dtype='i8', maxshape=(None,), chunks=True)
    W = g.create_dataset('W', (0,), dtype='b', maxshape=(None,), chunks=True)

    move_size = 0
    moves = 0
    games = 0
    game_size = 0

    # Loop through every game
    for game in read_games(fname_in):
        # Try to extract the game data
        extract_try = extract_gdata(game)
        if extract_try is not None:
            x, move_arr, winner = extract_try
        else:
            continue

        game_moves = len(move_arr)

        # Resize the h5 file if we're gonna need more space for boards and turns
        while moves + game_moves >= move_size:
            g.flush()
            move_size = 2 * move_size + 1
            print('Resizing board and moves to %s' % move_size)
            [d.resize(size=move_size, axis=0) for d in (X, M)]

        # Resize the h5 file if we're gonna need more space to store the winner
        while games + 1 >= game_size:
            g.flush()
            game_size = 2 * game_size + 1
            print('Resizing winners to %s' % game_size)
            W.resize(size=game_size, axis=0)

        X[moves:moves+game_moves] = x
        M[moves:moves+game_moves] = move_arr
        W[games] = winner

        moves += game_moves
        games += 1

    # Resize down to if we overshot
    print('Final Size for boards and moves: %s' % moves)
    [d.resize(size=moves, axis=0) for d in (X, M)]
    print('Final Size for winners: %s' % games)
    W.resize(size=games, axis=0)


class TestBoardMethods(tst.TestCase):

    def test_b2array(self):
        # Case 1: A start board
        start_board = 'r n b q k b n r' \
                      'p p p p p p p p' \
                      '. . . . . . . .' \
                      '. . . . . . . .' \
                      '. . . . . . . .' \
                      '. . . . . . . .' \
                      'P P P P P P P P' \
                      'R N B Q K B N R'
        expected =np.array([[-4, -2, -3, -5, -6, -3, -2, -4],
                            [-1, -1, -1, -1, -1, -1, -1, -1,],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 1, 1, 1, 1, 1, 1, 1, 1],
                            [ 4, 2, 3, 5, 6, 3, 2, 4]])
        np.testing.assert_array_equal(expected, b2array(start_board))

        # Case 2: A mid-game board
        mid_board = 'r . b q . . k r' \
                    'p p p . . . p p' \
                    '. . n b . . . .' \
                    '. . . p p . . .' \
                    '. . . . . . . .' \
                    '. . N P . N . .' \
                    'P P P . Q P P P' \
                    'R . B . K . . R'
        expected = np.array([[-4, 0, -3, -5, 0, 0, -6, -4],
                             [-1, -1, -1, 0, 0, 0, -1, -1],
                             [ 0, 0, -2, -3, 0, 0, 0, 0],
                             [ 0, 0, 0, -1, -1, 0, 0, 0],
                             [ 0, 0, 0, 0, 0, 0, 0, 0],
                             [ 0, 0, 2, 1, 0, 2, 0, 0],
                             [ 1, 1, 1, 0, 5, 1, 1, 1],
                             [ 4, 0, 3, 0, 6, 0, 0, 4]])
        np.testing.assert_array_equal(expected, b2array(mid_board))


if __name__ == '__main__':
    data_fn = '../ficsgamesdb_2016_standard2000_nomovetimes_1443264.pgn'
    to_h5(data_fn, '../chess_games_2016.h5')
    # tst.main()
