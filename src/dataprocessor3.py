import numpy as np
import h5py
import chess

CASTLE_KING = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0]])
CASTLE_MOVE = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 1, 0]])

class DataProcessor(object):

    def __init__(self, selection_labels=True, movement_labels=True, exclude_n=5, prev_boards=0, validation_split=0.15):
        # Set up some instance variables
        self.selection_labels = selection_labels
        self.movement_labels = movement_labels
        self.validation_split = validation_split
        self.prev_boards = prev_boards
        self.exclude_n= exclude_n
        self.VERBOSITY = 1

        # Initialize file stuff
        self.fn_ins = []
        self.num_games = 0
        self.num_moves = 0

        # Initialize all the dataset stuff
        self.X_turn = np.zeros((0, 8, 8), dtype=np.int8)
        self.X_moved = np.zeros((0, 8, 8), dtype=np.int8)
        self.train_inds = np.arange(0)
        self.val_inds = np.arange(0)
        self.train_size = 0
        self.val_size = 0

    def load_h5(self, fn_in):

        h5f = h5py.File(fn_in, 'r')
        print("Reading in file %s..." % fn_in)
        # Load the data
        X, M, W = [np.array(h5f[group], dtype=np.int8) for group in ('X', 'M', 'W')]

        # Make sure that we have the same number of boards as we do moves.
        assert(X.shape[0] == M.shape[0])
        num_moves = X.shape[0]
        print('%s moves in dataset' % num_moves)

        turn_inds = []
        moved_inds = []
        # Find the end game turns
        # BUG: Have to add 1 so end index is right... not sure why??
        end_inds = np.where(M == 0)[0] + 1

        # Make sure that we have the same number of ends to games as we do winners
        assert(len(end_inds) == len(W))
        num_games = len(end_inds)
        print('%s games in dataset' % num_games)

        print("Calculating turn and move inds")
        for i, winner in enumerate(W):
            # For now we'll disregard drawed games
            if winner == 0:

                # Mark for delete
                continue

            # Get the indices of the start and end of the game
            end_ind = end_inds[i]
            start_ind = 0 if i == 0 else end_inds[i-1] + 1

            # Get the indices of the winner's turn moves and post_move and add to array
            winner_moves = start_ind + np.where(M[start_ind:end_ind] == winner)[0]
            turn_inds.append(winner_moves)
            moved_inds.append(winner_moves + 1)

            # If black is the winner, we need to switch sides
            if winner == -1:
                X[start_ind:end_ind] = self.switch_sides(X[start_ind:end_ind]).astype(np.int8)

        # Build the new index array of game starts and ends
        assert(len(turn_inds) == len(moved_inds))
        print("Calculating start and end inds")
        start_inds = [0]
        end_inds = []
        for i in range(1, len(turn_inds)):
            start_inds.append(start_inds[i-1] + len(turn_inds[i-1]))
            end_inds.append(start_inds[i] - 1)
        end_inds.append(end_inds[-1] + len(moved_inds[-1]))
        assert(len(start_inds) == len(end_inds))
        num_games = len(end_inds)

        # Combine the indexes
        turn_inds = np.hstack(turn_inds)
        moved_inds = np.hstack(moved_inds)

        print("Extracting pre and post moves")
        X_turn = X[turn_inds]
        X_moved = X[moved_inds]
        num_moves = len(turn_inds)

        # Get the train and val indicies
        split_ind = int(num_games * (1-self.validation_split))
        print("Train-test Split Index: ", start_inds[split_ind])
        train_inds = np.arange(start_inds[split_ind])
        l = 0

        # Clean up the indicies
        for k, i in enumerate(train_inds):
            while i > end_inds[l]:
                l += 1
            if i - start_inds[l] < max(int(self.prev_boards / 2) + 1, self.exclude_n):
                # Mark for delete
                train_inds[k] = -1
        train_inds = train_inds[train_inds >= 0]
        # print(train_inds)


        val_inds = np.arange(start_inds[split_ind], X_turn.shape[0])
        l = split_ind

        # Clean up the indicies
        for k, i in enumerate(val_inds):
            while i > end_inds[l]:
                l += 1
            if i - start_inds[l] < max(int(self.prev_boards / 2) + 1, self.exclude_n):
                # Mark for delete
                val_inds[k] = -1
        val_inds = val_inds[val_inds >= 0]

        # Finished processing, now add it to the instance variables
        # Update the indicies
        self.train_inds = np.concatenate([self.train_inds, train_inds + len(self.X_turn)])
        self.val_inds = np.concatenate([self.val_inds, val_inds + len(self.X_turn)])

        # Update the dataset
        self.X_turn = np.concatenate([self.X_turn, X_turn])
        self.X_moved = np.concatenate([self.X_moved, X_moved])

        # Some sanity checks
        assert(self.X_turn.shape == self.X_moved.shape)

        # Update the train and val size
        self.train_size = len(self.train_inds)
        self.val_size = len(self.val_inds)

        # Update the dataset metas
        self.num_games += len(start_inds)
        self.num_moves = len(self.X_turn)

        self.fn_ins.append(fn_in)

    def array2b(self, arr):
        """
        Turns an 8 x 8 x 12 array into Board object with white's Turns. Useful
        for testing and visualizing

        Arguments:
        arr -- the array to turn into a board

        Returns:
        A Board object that has the position of the pieces encoded, but no other
        info.
        """
        if (arr.ndim == 2):
            arr = self.split_boards(arr[np.newaxis])[0]

        board = chess.Board(fen=None)
        for p in range(1,7):
            bslice = np.flipud(arr[:, :, p-1])
            for s in np.flatnonzero(bslice):
                board.set_piece_at(s, chess.Piece(p, True))

            bslice = np.flipud(arr[:, :, p+5])
            for s in 1*(np.flatnonzero(bslice) - 64):
                board.set_piece_at(s, chess.Piece(p, False))
        return board

    @staticmethod
    def switch_sides(X_game):
        # Flip the boards and invert the pieces
        return -1 * X_game[:, ::-1, ...]

    @staticmethod
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


    def get_labels(self, x_before, x_after, debug=False):
        """
        Takes the winner boards pre-move and post-move
        and returns a tuple with the following elements
            1) a 1 at the position of the piece selected to move (0's elsewhere)
            2) a 1 at the position where the piece was moved to (0's elsewhere)
        """
        # Find the residual boards
        x_residual = x_before - x_after
        # Use the residual boards to find the selected piece
        selection = (x_residual > 0).astype(np.float32)
        # And where it moved to
        movement = (x_residual < 0).astype(np.float32)

        for i in range(len(x_residual)):
            if debug and not (np.sum(selection[i]) in {1., 2.}):
                print("Residual Sum incorrect:")
                print("Turn board:")
                print(self.array2b(x_before[i]))
                print("Move board:")
                print(self.array2b(x_after[i]))
                print("Residual Board:")
                print(x_residual[i])

                raise ValueError
            # If the move was a castle, normalize output
            if np.sum(selection[i]) == 2.:
                selection[i] *= CASTLE_KING

            if debug and not (np.sum(movement[i]) in {1., 2.}):
                print("Residual Sum incorrect:")
                print("Turn board:")
                print(x_before[i])
                print("Move board:")
                print(x_after[i])
                print("Residual Board:")
                print(x_residual[i])

                raise ValueError
            # If the move was a castle, normalize output
            if np.sum(movement[i]) == 2:
                movement[i] *= CASTLE_MOVE

        return selection, movement

    def select_labels(self, selections, movements):
        """
        Makes labels for keras NN based on which are specified
        """
        y = []
        if self.selection_labels:
            y.append(selections)
        if self.movement_labels:
            y.append(movements)
        if len(y) == 1:
            y = y[0]
        return y

    def chessgen(self, batch_size=32, shuffle=True, validation=False, debug=False):
        """
        A generator for a keras NN
        Yields a tuple of boards
        if prev_boards is true, then the dataset should NOT be shuffled upon input
        """
        inds = self.val_inds if validation else self.train_inds
        print(inds)
        # print("Entered Generator")
        while True:
            # Shuffle if we need to
            if shuffle:
                np.random.shuffle(inds)
            # print("Starting epoch")
            for i in range(0, len(inds), batch_size):
                # print("Preparing new batch")
                # Get the batch of boards
                x_batch = self.X_turn[inds[i:i+batch_size]]
                # if debug: print('Board pre-move:\n{}'.format(x_batch[0]))
                # Get the labels for the boards
                x_moved_batch = self.X_moved[inds[i:i+batch_size]]
                selections, movements = self.get_labels(x_batch, x_moved_batch)
                # if debug: print('Board post-move:\n{}'.format(x_moved_batch[0]))
                # Split the boards into the different piece boards
                x_batch = self.split_boards(x_batch)

                # Get all the previous boards and split them
                # print("Creating previous boards")
                if self.prev_boards:
                    num_prev_turn = int(self.prev_boards / 2)
                    num_prev_move = int((self.prev_boards + 1) / 2)
                    # Resize the batch to fit the prev boards
                    m = 12 * (num_prev_turn + 1)
                    n = 12 * self.prev_boards
                    x_batch = np.concatenate([x_batch, np.empty(x_batch.shape[:3] + (n,))], axis=3)

                    for j, k in enumerate(range(12, m, 12)):
                        prev_inds = inds[i:i+batch_size] - (j + 1)
                        x_batch[:, ..., k: k+12] = self.split_boards(self.X_turn[prev_inds])

                    for j, k in enumerate(range(m, n + 12, 12)):
                        prev_inds = inds[i:i+batch_size] - (j + 1)
                        x_batch[:, ..., k: k+12] = self.split_boards(self.X_moved[prev_inds])

                    if debug:
                        for b in range(batch_size):
                            print("Verifying past moves %s" %b)
                            for j, k in enumerate(range(0, 12 * (self.prev_boards + 1), 12)):
                                if j == 0:
                                    print("Current board:")
                                    print(self.array2b(x_batch[b,..., k:k+12]))
                                elif j <= num_prev_turn:
                                    print("Turn prev board:")
                                    print(self.array2b(x_batch[b,..., k:k+12]))
                                else:
                                    print("Move prev board:")
                                    print(self.array2b(x_batch[b,..., k:k+12]))
                            input("Continue?")

                # Make the labels
                # if self.VERBOSITY >= 1: print("Making Labels")
                y_batch = self.select_labels(selections, movements)

                yield x_batch, y_batch
