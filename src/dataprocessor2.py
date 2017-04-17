from parse_games import read_games
import chess
import parse_games as pg
import numpy as np
from dataprocessor import split_boards
import unittest as tst
from multiprocessing import Pool

CASTLE_KING = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0]])
QUEEN_CASTLE = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0]])
KING_CASTLE = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0]])

def make_gn_arr(gn):
    results = {'1-0': True, '0-1': False, '1/2-1/2': None}
    gn_arr = []
    # Ensure we are at the root of the game
    if not gn.starts_variation():
        gn = gn.root()
    # Get the winner of the game
    winner = results[gn.headers["Result"]]
    # If tie, throw out the game
    if winner is None:
        return gn_arr
    turn = True
    while gn.variations:
        # Go to the next move
        gn = gn.variations[0]
        turn = not turn
        if winner == turn:
            gn_arr.append(gn)
    return gn_arr

def unwind_games(pgn_fn):
    """
    Unwinds each game into its moves and puts them into
    a flattened array.

    Arguments:
    pgn_fn -- The filepath of the pgn file to read

    Returns:
    An array filled with GameNode objects for each move in each game
    """
    # pool = Pool(processes=8)
    gn_arrs = []
    for i, gn_arr in enumerate(map(make_gn_arr, read_games(pgn_fn))):
        if i % 200 == 0: print("Read %s games" % i)
        if i == 1000: break
        gn_arrs.append(gn_arr)
    return np.hstack(gn_arrs)

def squares2array(square_set):
    """
    Turns a SquareSet object into an 8 x 8 numpy array

    Arguments:
    square_set -- The SquareSet object denoting which squares were selected

    Returns:
    An 8 x 8 numpy array with 1's in the selected squares and 0 Otherwise
    """
    return np.fliplr((np.fromstring(format(square_set.mask, '064b'), np.int8) - 48).reshape((8, 8)))

def b2array(board):
    """
    Turns a Board object into an 8 x 8 x 12 numpy array with the positions
    of each piece

    Arguments:
    board -- A Board object with the current board state

    Returns:
    An 8 x 8 x 12 numpy array where each channel represents a pieces'
    locations on an 8 x 8 chess board
    """
    ret_arr = np.zeros((8, 8, 12), np.int8)
    for c in range(2):
        for p in range(1,7):
            ind = c * 6 + (p - 1)
            # We want to start with white first for consistency (not c)
            ret_arr[:, :, ind] = squares2array(board.pieces(p, not c))
    return ret_arr

def array2b(arr):
    """
    Turns an 8 x 8 x 12 array into Board object with white's Turns. Useful
    for testing and visualizing

    Arguments:
    arr -- the array to turn into a board

    Returns:
    A Board object that has the position of the pieces encoded, but no other
    info.
    """
    board = chess.Board(fen=None)
    for p in range(1,7):
        bslice = np.flipud(arr[:, :, p-1])
        for s in np.flatnonzero(bslice):
            board.set_piece_at(s, chess.Piece(p, True))

        bslice = np.flipud(arr[:, :, p+5])
        for s in 1*(np.flatnonzero(bslice) - 64):
            board.set_piece_at(s, chess.Piece(p, False))
    return board

def past_moves(gn, num_past):
    """
    Turns a game node into a 3d numpy array of its current move
    and num_past - 1 past moves

    Arguments:
    gn -- The game node in question
    num_past -- The number of past moves to generate

    Returns:
    A 3d numpy array with shape 8 x 8 x num_past with numpy arrays of
    the past moves
    """
    past_move_arr = np.empty((8, 8, num_past * 12), np.int8)
    # print('before: ', past_move_arr.shape)
    for i in range(num_past):
        # Figure out a more efficient way translate to array
        # print(gn.board())
        past_move_arr[:, :, i*12: (i+1)*12] = b2array(gn.board())
        if gn.parent is not None:
            gn = gn.parent
    # print('after: ', past_move_arr.shape)
    return past_move_arr

def board_pieces(board, color):
    """
    Turns a board object into a squareset wherever a piece of
    a specific color occurs

    Arguments:
    board -- The current board position
    color -- boolean for color of piece, true if white, false for black

    Returns:
    A squareset that has data on the positions of all of the pieces
    """
    return chess.SquareSet(board.occupied_co[color])

def square_gen(square_set):
    """
    Generates all the square values for the squares
    included in the SquareSet object

    Arguments:
    square_set -- A SquareSet object with the wanted squares

    Returns:
    A generator that yields the index id of each square we wanted
    """
    return (i for i in range(64) if format(square_set.mask, '064b')[-i-1] == '1')

def capture(board, threat=True):
    """
    Creates an 8 x 8 x 1 array with 1's on a square where piece can
    capture an enemy piece (or pieces that are threatened if threat)

    Arguments:
    board -- The current game board
    threat -- Whether to find threatened pieces or pieces that can capture

    Returns:
    An 8 x 8 x 1 array with 1's where identified pieces are
    """
    ret_arr = np.zeros((64,), np.int8)
    for s in square_gen(board_pieces(board, board.turn)):
        if threat:
            attacked = board.attackers(not board.turn, s)
        else:
            enem_squares = board_pieces(board, not board.turn)
            attacked = enem_squares & board.attacks(s)
        ret_arr[s] = 1 if attacked else 0
    return np.flipud(ret_arr.reshape((8, 8, 1)))

def pinned(board):
    """
    Creates an 8 x 8 x 1 array with 1's on a square where the player's piece
    is pinned

    Arguments:
    board -- The current game board

    Returns:
    An 8 x 8 x 1 array with 1's where identified pieces are
    """
    enem_squares = board_pieces(board, not board.turn)
    ret_arr = np.zeros((64,), np.int8)
    for enem_s in square_gen(enem_squares):
        attacked_pieces = board_pieces(board, board.turn) & board.attacks(enem_s)
        for s in square_gen(attacked_pieces):
            if board.is_pinned(board.turn, s):
                ret_arr[s] = 1
                break
    return np.flipud(ret_arr.reshape((8, 8, 1)))

def threat_set(board, s):
    """
    Creates a SquareSet of squares where current player's pieces are
    threatened by the piece at s

    Arguments:
    board -- the current game board state
    s -- the square of the enemy piece

    Returns:
    A SquareSet with 1's where the player's pieces are threatened
    """
    return board.attacks(s) & board_pieces(board, board.turn)

def is_fork(attacked_pieces):
    """
    Tells whether the attacking piece is forking something

    Arguments:
    attacked_pieces -- The SquareSet of pieces being attacked
                       by the piece in question

    Returns:
    A boolean denoting whether a fork or not
    """
    # If the mask is a power of 2 or 0, it is not a fork
    return not is_power2(attacked_pieces)

def pin_check(board, attacked_pieces):
    """
    Tells whether the attacking piece is pinning another

    Arguments:
    attacked_pieces -- The SquareSet of pieces being attacked
                       by the piece in question
    board -- The current game board state

    Returns:
    A boolean denoting whether one of the attacked pieces is pinned
    """
    # Check all of the pieces being threatened
    for s in square_gen(attacked_pieces):
        if board.is_pinned(board.piece_at(s).color, s):
            return True

def is_power2(x):
    return not x.mask & (x.mask - 1)

def create_filters(gn, num_past, piece=True, moves=False):

    pfilt = np.zeros((8, 8, 7 + num_past*12), dtype=np.int8)
    pfilt[:, :, 7:] = past_moves(gn, num_past)
    board = gn.board()
    pfilt[:, :, 5:6] = capture(board, threat=True)
    pfilt[:, :, 6:7] = pinned(board)
    for l in board.legal_moves:
        from_ind = np.unravel_index(l.from_square, (8,8))
        from_ind = (7 - from_ind[0], from_ind[1])
        # Check if the move is a capture
        if board.is_capture(l):
            pfilt[from_ind[0], from_ind[1], 0] = 1
        # Our next checks are with the board if we were to make the move
        board.push(l)
        # Check if the move puts the enemy in check
        if board.is_check():
            pfilt[from_ind[0], from_ind[1], 1] = 1
        # Find all of enemy's pieces threatened by move l
        attacked_pieces = threat_set(board, l.to_square)
        # Check for threats to enemy
        if attacked_pieces:
            pfilt[from_ind[0], from_ind[1], 2] = 1
        # Check for forks
        # If the mask is a power of 2 or 0, it is not a fork
        if is_fork(attacked_pieces):
            pfilt[from_ind[0], from_ind[1], 3] = 1
        # Check for pins
        if pin_check(board, attacked_pieces):
            pfilt[from_ind[0], from_ind[1], 4] = 1
        # Reset the board to the original
        board.pop()
    return pfilt

def unsplit(X_boards):
    return sum((i + 1 if i < 6 else -1 * i - 1) * X_boards[:, :, :, i] for i in range(12))

def switch_sides(X_game, prev_filts=0):
    # Flip the boards and swap the black and white filters
    flipped = X_game[::-1, ...]
    # print(flipped.shape)
    tmp = None
    for i in range(prev_filts, flipped.shape[-1], 12):
        flipped[:, :, i:i+12] = split_boards(-1 * unsplit(flipped[np.newaxis, :, :, i:i+12]), split=12)[0]
    return flipped

def get_labels(X_before, X_after, debug=False):
    """
    Takes the winner boards pre-move and post-move
    and returns a tuple with the following elements
        1) a 1 at the position of the piece selected to move (0's elsewhere)
        2) a 1 at the position where the piece was moved to (0's elsewhere)
    """
    # Find the residual boards
    X_residual = unsplit(X_before) - unsplit(X_after)
    # Use the residual boards to find the selected piece
    selection = (X_residual > 0).astype(np.float32)
    # print(selection)
    # Get the number of pieces moved
    num_moves = np.sum(selection, axis=(1, 2))
    assert(len(num_moves.shape) == 1)
    # If the move was a castle, force it to select the king
    selection[num_moves == 2.] = CASTLE_KING
    if np.any(num_moves > 2. or num_moves <= 0):
        raise ValueError("Multiple moves between both boards")

    # And where it moved to
    movement = (X_residual < 0).astype(np.float32)
    # Get the number of pieces moved
    num_moves = np.sum(selection, axis=(1, 2))
    assert(len(num_moves.shape) == 1)
    # Clean up the Castle moves
    for i in range(len(num_moves)):
        if num_moves[i] == 2:
            movement[i] = QUEEN_CASTLE if movement[i, 7, 2] == 1 else KING_CASTLE
    # If the move was a castle, normalize output
    if np.any(num_moves > 2. or num_moves <= 0):
        raise ValueError("Multiple moves between both boards")
    return selection, movement

def select_labels(selections, movements, selection_labels, movement_labels):
    """
    Makes labels for keras NN based on which are specified
    """
    y = []
    if selection_labels:
        y.append(selections)
    if movement_labels:
        y.append(movements)
    if len(y) == 1:
        y = y[0]
    return y

def batch_gen(pgn_fn, num_past=8, selections=True, movements=True, batch_size=32, shuffle=True):
    game_boards = unwind_games(pgn_fn)
    if shuffle:
        inds = np.random.permutation(len(game_boards))
    for i in range(0, len(game_boards), batch_size):
        batch_inds = inds[i:i+batch_size]
        x_batch = np.empty((len(batch_inds), 8, 8, 7 + num_past*12))
        x_after = np.empty((len(batch_inds), 8, 8, 12))
        # Fill in the batch
        for ind, j in enumerate(batch_inds):
            x_batch[ind] = create_filters(game_boards[j], num_past)
            x_after[ind] = past_moves(game_boards[j].variations[0], 1)
            if not game_boards[j].board().turn:
                # TODO carry over implementation of switch sides
                print("Switching Sides")
                x_batch[ind] = switch_sides(x_batch[ind], prev_filts=7)
                x_after[ind] = switch_sides(x_after[ind], prev_filts=0)
            print("Next Move:\n", array2b(x_after[ind]))

        # TODO carry over implementation of label making
        selection, movement = get_labels(x_batch[:, :, :, 7:7+12], x_after)
        y_batch = select_labels(selection, movement, selection_labels=selections, movement_labels=movements)

        yield x_batch, y_batch


class TestBoardMethods(tst.TestCase):

    def test_b2array(self):
        test1 = chess.Board()
        expected = split_boards(pg.b2array(str(test1))[np.newaxis, :, :], split=12)[0]
        np.testing.assert_array_equal(expected, b2array(test1))
        self.assertEqual(str(array2b(b2array(test1))), str(test1))

        test2 = chess.Board("rnb1k2r/ppp2ppp/5n2/3q4/1b1P4/2N5/PP3PPP/R1BQKBNR w KQkq - 3 7")
        expected = split_boards(pg.b2array(str(test2))[np.newaxis, :, :], split=12)[0]
        np.testing.assert_array_equal(expected, b2array(test2))
        self.assertEqual(str(array2b(b2array(test2))), str(test2))

    def test_pastmoves(self):
        pgn_fn = '../ficsgamesdb_2015_CvC_nomovetimes_1443974.pgn'
        game = None
        for gn in read_games(pgn_fn):
            game = gn
            break
        game = game.end()
        num_past = 8
        out = past_moves(game, num_past)
        # print(out.transpose(2, 0, 1))

        for i in range(num_past):
            board = game.board()
            past_move_arr = out[:, :, i*12:(i+1)*12]
            # print(past_move_arr.shape)
            # print(board)
            # print('\n')
            # print(array2b(past_move_arr))
            self.assertEqual(str(array2b(past_move_arr)), str(board))
            game = game.parent

    def test_capture(self):
        test1 = chess.Board()
        expected = np.zeros((8, 8, 1))
        np.testing.assert_array_equal(expected, capture(test1))

        test2 = chess.Board("rnb1k2r/ppp2ppp/5n2/3q4/1b1P4/2N5/PP3PPP/R1BQKBNR w KQkq - 3 7")
        print("Testing Threats")
        print("White" if test2.turn else "Black")
        print(test2)
        print(test2.turn)
        print(capture(test2).transpose(2, 0, 1))

    def test_pinned(self):
        test1 = chess.Board()
        expected = np.zeros((8, 8, 1))
        np.testing.assert_array_equal(expected, pinned(test1))

        test2 = chess.Board("rnb1k2r/ppp2ppp/5n2/3q4/1b1P4/2N5/PP3PPP/R1BQKBNR w KQkq - 3 7")
        print("Testing Pinned")
        print("White" if test2.turn else "Black")
        print(test2)
        print(pinned(test2).transpose(2, 0, 1))

    def test_other_filters(self):
        test1 = chess.Board()
        pgn_fn = '../ficsgamesdb_2015_CvC_nomovetimes_1443974.pgn'
        game = None
        for gn in read_games(pgn_fn):
            game = gn
            break
        game = game.end()
        for i in range(41):
            game = game.parent
        filts = create_filters(game, 8)
        tests = ["Captures", "Check", "Attack", "Fork", "Pinning", "Threat", "Pinned"]
        print("White" if game.board().turn else "Black")
        print(game.board())
        for i, test in enumerate(tests):
            print("Testing " + test)
            print(filts[:, :, i])

    def test_batch_gen(self):
        pgn_fn = '../ficsgamesdb_2015_CvC_nomovetimes_1443974.pgn'
        bg = batch_gen(pgn_fn, 8, batch_size=1)
        i = 0
        for x, (y_sel, y_mov) in bg:
            print("Original Board:\n", array2b(x[0, :, :, 7:7+12]))
            print("Selection:\n",y_sel[0].astype(np.int8))
            print("Movement:\n",y_mov[0].astype(np.int8))
            i += 1
            if i == 10: break


if __name__ == "__main__":
    tst.main()
