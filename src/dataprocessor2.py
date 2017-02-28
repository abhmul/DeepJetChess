from parse_games import read_games
import numpy as np

def unwind_games(pgn_fn):
    """
    Unwinds each game into its moves and puts them into
    a flattened array.

    Arguments:
    pgn_fn -- The filepath of the pgn file to read

    Returns:
    An array filled with GameNode objects for each move in each game
    """
    gn_arr = []
    for gn in read_games(pgn_fn):
        # Ensure we are at the root of the game
        if not gn.starts_variation():
            gn = gn.root()
        while gn.variations:
            gn_arr.append(gn)
            # Go to the next move
            gn = gn.variations[0]
    return np.array(gn_arr)

def squares2array(square_set):
    """
    Turns a SquareSet object into an 8 x 8 numpy array

    Arguments:
    square_set -- The SquareSet object denoting which squares were selected

    Returns:
    An 8 x 8 numpy array with 1's in the selected squares and 0 Otherwise
    """
    return (np.fromstring(bin(square_set.mask)[2:], np.int8) - 48).reshape((8, 8))

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
    for c in range(1):
        for p in range(1,7):
            ind = c * 6 + (p - 1)
            ret_arr[:, :, ind] = squares2array(board.pieces(p, c))
    return ret_arr

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
    for i in range(num_past):
        # Figure out a more efficient way translate to array
        past_move_arr[:, :, i*12: (i+1)*12] = b2array(gn.board())
        if gn.parent is not None:
            gn = gn.parent
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
    return board.occupied_co[color]

def square_gen(square_set):
    """
    Generates all the square values for the squares
    included in the SquareSet object

    Arguments:
    square_set -- A SquareSet object with the wanted squares

    Returns:
    A generator that yields the index id of each square we wanted
    """
    return (i for i in range(64) if bin(square_set.mask)[2:][i] == '1')

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
    enem_squares = board_pieces(board, not board.turn)
    ret_arr = np.zeros((64,), np.int8)
    for s in square_gen(board_pieces(board, board.turn)):
        if threat:
            attacked = enem_squares & board.attackers(s)
        else:
            attacked = enem_squares & board.attacks(s)
        ret_arr[s] = 1 if attacked else 0
    return ret_arr.reshape((8, 8, 1))

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
    return ret_arr.reshape((8, 8, 1))

def is_power2(x):
    return not x & (x - 1)

def create_filters(gn, num_past, piece=True, moves=False):

    pfilt = np.zeros((8, 8, 7 + num_past), dtype=np.int8)
    pfilt[:, :, 7:] = past_moves(gn, num_past)
    board = gn.board()
    pfilt[:, :, 5] = capture(board, threat=True)
    pfilt[:, :, 6] = pinned(board)
    for l in board.legal_moves:
        # Check if the move is a capture
        if board.is_capture(l):
            pfilt[np.unravel_index(l.from_square), 0] = 1
        # Our next checks are with the board if we were to make the move
        board.push(l)
        # Check if the move puts the enemy in check
        if board.is_check():
            pfilt[np.unravel_index(l.from_square), 1] = 1
        # Find all the pieces attacked by that move
        attacked_pieces = board.attacks(l.to_square) & board_pieces(board, board.turn)
        # Check for threats to enemy
        if attacked_pieces:
            pfilt[np.unravel_index(l.from_square), 2] = 1
        # Check for forks
        # If the mask is a power of 2 or 0, it is not a fork
        if not is_power2(attacked_pieces):
            pfilt[np.unravel_index(l.from_square), 3] = 1
        # Check for pins
        for s in square_gen(attacked_pieces):
            if board.is_pinned(board.turn, s):
                pfilt[np.unravel_index(l.from_square), 4] = 1
                break
        # Reset the board to the original
        board.pop()
    return pfilt
