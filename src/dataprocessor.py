import numpy as np
import h5py

import unittest as tst

def switch_sides(X_game):
    # Flip the boards and invert the pieces
    return -1 * X_game[:, ::-1, ...]


def load_h5(fn_in, shuffle=True, exclude_n=0, random_seed=None):
    """
    Splits the board array into one for the winner and one for post-move
    """
    if random_seed is not None:
        np.random_seed(random_seed)
    h5f = h5py.File(fn_in, 'r')
    # Load the data
    X, M, W = [h5f[group] if group == 'X' else h5f[group][()] for group in ('X', 'M', 'W')]
    # Make sure that we have the same number of boards as we do moves.
    assert(X.shape[0] == M.shape[0])
    print('%s moves in dataset' % X.shape[0])
    turn_inds = []
    moved_inds = []
    # Find the end game turns
    end_inds = np.where(M == 0)[0]
    # Make sure that we have the same number of ends to games as we do winners
    assert(len(end_inds) == len(W))
    print('%s games in dataset' % len(end_inds))
    for i, winner in enumerate(W):
        # For now we'll disregard drawed games
        if winner == 0:
            continue
        # Get the indices of the start and end of the game
        end_ind = end_inds[i]
        start_ind = 0 if i == 0 else end_inds[i-1] + 1
        # Get the indices of the winner's turn moves and post_move and add to array
        winner_moves = start_ind + np.where(M[start_ind:end_ind] == winner)[0]
        # Exclude the first exclude_n moves
        if exclude_n > 0:
            winner_moves = winner_moves[exclude_n:]
            # If the game did not last longer than the first
            # exclude_n moves, skip it
            if len(winner_moves) == 0:
                continue
            # Modify the start ind to not include the excluded moves
            start_ind = winner_moves[0]
        turn_inds.append(winner_moves)
        moved_inds.append(winner_moves + 1)
        # If black is the winner, we need to switch sides
        if winner == -1:
            X[start_ind:end_ind] = switch_sides(X[start_ind:end_ind])
    # Combine the indexes
    turn_inds = np.hstack(turn_inds)
    moved_inds = np.hstack(moved_inds)
    # Make sure that the inds for when the winner is to move and after are the same length
    assert(len(turn_inds) == len(moved_inds))
    if shuffle:
        shuffle_inds = np.arange(len(turn_inds))
        np.random.shuffle(shuffle_inds)
        turn_inds = turn_inds[shuffle_inds]
        moved_inds = moved_inds[shuffle_inds]

    return X[turn_inds], X[moved_inds]

def get_labels(X_before, X_after):
    """
    Takes the winner boards pre-move and post-move
    and returns a tuple with the following elements
        1) a 1 at the position of the piece selected to move (0's elsewhere)
        2) a 1 at the position where the piece was moved to (0's elsewhere)
    """
    # Find the residual boards
    X_residual = X_before - X_after
    # Use the residual boards to find the selected piece
    selection = (X_residual > 0).astype(np.float32)
    # And where it moved to
    movement = (X_residual < 0).astype(np.float32)
    return selection, movement

def split6(X_board):
    """
    Turns the boards into 6 new boards for each piece
    Player is 1 and enemy is -1
    """
    X = np.empty(X_board.shape[:3] + (6,))
    for pieceval in range(1,7):
        # Select the squares with the specified piece
        X[:, :, :, pieceval-1] = ((X_board == pieceval).astype(np.float32) + -1 * (X_board == -1*pieceval).astype(np.float32))
    return X

def split12(X_board):
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

def split_boards(X_board, split):
    """
    Splits each board into the specified number of channels
    """
    if split == 6:
        return split6(X_board)
    elif split == 12:
        return split12(X_board)
    else:
        raise ValueError("Split channels should either be 6 or 12. Given %s" % split)

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


def chessgen(X_before, X_after, selection_labels=True, movement_labels=True, split=12, batch_size=32, shuffle=True, debug=False):
    """
    A generator for a keras NN
    Yields a tuple of boards
    """
    if debug:
        batch_size = 1
    inds = np.arange(X_before.shape[0])
    while True:
        # Shuffle if we need to
        if shuffle:
            np.random.shuffle(inds)
        for i in range(0, len(inds), batch_size):
            # Get the batch of boards
            x_batch = X_before[inds[i:i+batch_size]]
            if debug: print('Board pre-move:\n{}'.format(x_batch[0]))
            # Get the labels for the boards
            selections, movements = get_labels(x_batch, X_after[inds[i:i+batch_size]])
            if debug: print('Board post-move:\n{}'.format(X_after[inds[i:i+batch_size]][0]))
            # Split the boards into the different piece boards
            x_batch = split_boards(x_batch, split)
            # Make the labels
            y_batch = select_labels(selections, movements, selection_labels, movement_labels)

            yield x_batch, y_batch

class TestProcessingMethods(tst.TestCase):

    def test_switch_sides(self):
        boards = np.array([[[-4, -2, -3, -5, -6, -3, -2, -4],
                            [-1, -1, -1, -1, -1, -1, -1, -1,],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 1, 1, 1, 1, 1, 1, 1, 1],
                            [ 4, 2, 3, 5, 6, 3, 2, 4]],
                           [[-4, 0, -3, -5, 0, 0, -6, -4],
                            [-1, -1, -1, 0, 0, 0, -1, -1],
                            [ 0, 0, -2, -3, 0, 0, 0, 0],
                            [ 0, 0, 0, -1, -1, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 2, 1, 0, 2, 0, 0],
                            [ 1, 1, 1, 0, 5, 1, 1, 1],
                            [ 4, 0, 3, 0, 6, 0, 0, 4]]])
        expected = np.empty((2, 8, 8))
        expected[0] = boards[0]
        expected[1] = np.array([[-4, 0, -3, 0, -6, 0, 0, -4],
                                [-1, -1, -1, 0, -5, -1, -1, -1],
                                [ 0, 0, -2, -1, 0, -2, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 1, 1, 0, 0, 0],
                                [ 0, 0, 2, 3, 0, 0, 0, 0],
                                [ 1, 1, 1, 0, 0, 0, 1, 1],
                                [ 4, 0, 3, 5, 0, 0, 6, 4]])
        np.testing.assert_array_equal(expected, switch_sides(boards))

    def test_get_labels(self):
        boards = np.array([[[-4, -2, -3, -5, -6, -3, -2, -4],
                            [-1, -1, -1, -1, -1, -1, -1, -1,],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 1, 1, 1, 1, 1, 1, 1, 1],
                            [ 4, 2, 3, 5, 6, 3, 2, 4]],
                           [[-4, 0, -3, -5, 0, 0, -6, -4],
                            [-1, -1, -1, 0, 0, 0, -1, -1],
                            [ 0, 0, -2, -3, 0, 0, 0, 0],
                            [ 0, 0, 0, -1, -1, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 2, 1, 0, 2, 0, 0],
                            [ 1, 1, 1, 0, 5, 1, 1, 1],
                            [ 4, 0, 3, 0, 6, 0, 0, 4]]])

        afters = np.array([[[-4, -2, -3, -5, -6, -3, -2, -4],
                            [-1, -1, -1, -1, -1, -1, -1, -1,],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 1, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 1, 1, 1, 1, 0, 1, 1, 1],
                            [ 4, 2, 3, 5, 6, 3, 2, 4]],
                           [[-4, 0, -3, -5, 0, 0, -6, -4],
                            [-1, -1, -1, 0, 0, 0, -1, -1],
                            [ 0, 0, -2, -3, 0, 0, 0, 0],
                            [ 0, 0, 0, 2, -1, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 1, 0, 2, 0, 0],
                            [ 1, 1, 1, 0, 5, 1, 1, 1],
                            [ 4, 0, 3, 0, 6, 0, 0, 4]]])

        expected_selections = np.zeros((2, 8, 8), dtype=np.float32)
        expected_selections[0, 6, 4] = 1.
        expected_selections[1, 5, 2] = 1.
        expected_movements = np.zeros((2, 8, 8), dtype=np.float32)
        expected_movements[0, 4, 4] = 1.
        expected_movements[1, 3, 3] = 1.

        selections, movements = get_labels(boards, afters)
        np.testing.assert_array_equal(expected_selections, selections)
        np.testing.assert_array_equal(expected_movements, movements)

    def test_split(self):
        boards = np.array([[[-4, -2, -3, -5, -6, -3, -2, -4],
                            [-1, -1, -1, -1, -1, -1, -1, -1,],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 1, 1, 1, 1, 1, 1, 1, 1],
                            [ 4, 2, 3, 5, 6, 3, 2, 4]],
                           [[-4, 0, -3, -5, 0, 0, -6, -4],
                            [-1, -1, -1, 0, 0, 0, -1, -1],
                            [ 0, 0, -2, -3, 0, 0, 0, 0],
                            [ 0, 0, 0, -1, -1, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 2, 1, 0, 2, 0, 0],
                            [ 1, 1, 1, 0, 5, 1, 1, 1],
                            [ 4, 0, 3, 0, 6, 0, 0, 4]]])

        expected6 = np.zeros((2, 6, 8, 8))
        expected12 = np.zeros((2, 12, 8, 8))
        expected12[0, 0, 6] = np.ones(8)
        expected12[0, 1, 7, 1] = 1
        expected12[0, 1, 7, 6] = 1
        expected12[0, 2, 7, 2] = 1
        expected12[0, 2, 7, 5] = 1
        expected12[0, 3, 7, 0] = 1
        expected12[0, 3, 7, 7] = 1
        expected12[0, 4, 7, 3] = 1
        expected12[0, 5, 7, 4] = 1

        expected12[0, 6, 1] = np.ones(8)
        expected12[0, 7, 0, 1] = 1
        expected12[0, 7, 0, 6] = 1
        expected12[0, 8, 0, 2] = 1
        expected12[0, 8, 0, 5] = 1
        expected12[0, 9, 0, 0] = 1
        expected12[0, 9, 0, 7] = 1
        expected12[0, 10, 0, 3] = 1
        expected12[0, 11, 0, 4] = 1

        expected6[0, 0, 6] = np.ones(8)
        expected6[0, 1, 7, 1] = 1
        expected6[0, 1, 7, 6] = 1
        expected6[0, 2, 7, 2] = 1
        expected6[0, 2, 7, 5] = 1
        expected6[0, 3, 7, 0] = 1
        expected6[0, 3, 7, 7] = 1
        expected6[0, 4, 7, 3] = 1
        expected6[0, 5, 7, 4] = 1

        expected6[0, 0, 1] = -1 * np.ones(8)
        expected6[0, 1, 0, 1] = -1
        expected6[0, 1, 0, 6] = -1
        expected6[0, 2, 0, 2] = -1
        expected6[0, 2, 0, 5] = -1
        expected6[0, 3, 0, 0] = -1
        expected6[0, 3, 0, 7] = -1
        expected6[0, 4, 0, 3] = -1
        expected6[0, 5, 0, 4] = -1

        expected12 = expected12.transpose(0, 2, 3, 1)
        expected6 = expected6.transpose(0, 2, 3, 1)

        np.testing.assert_array_equal(expected12[0], split_boards(boards, 12)[0])
        np.testing.assert_array_equal(expected6[0], split_boards(boards, 6)[0])


if __name__ == '__main__':
    tst.main()
