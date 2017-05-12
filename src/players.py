import time

import chess
import chess.pgn
import sunfish

from np_board_utils import sb2array, NpBoard, WIN, LOSS

MAXD = 1

class Player(object):
    def move(self, gn_current):
        raise NotImplementedError()


class Computer(Player):
    def __init__(self, comparator, maxd=MAXD):
        self._comparator = comparator
        self._gn = None
        self._maxd = maxd

    def move(self, gn_current):
        assert(gn_current.board(_cache=True).turn == True)

        self._gn = gn_current

        alpha = LOSS
        beta = WIN

        depth = self._maxd
        t0 = time.time()
        # Board in array format
        # Move in python-chess format
        best_board, best_move = self.alphabeta(self._gn, self._comparator, depth, LOSS, WIN, True)
        print("Depth %s : Move %s : Time %s" %(depth, best_move, time.time() - t0))

        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = best_move

        return gn_new

    @staticmethod
    def alphabeta(gn_current, comparator, depth, alpha, beta, maximizing_player):
        """
        Runs position alpha-beta pruning using comparator to compare positions
        Arguments:
            gn_current -- The current game node (pgn python-chess format)
            comparator -- The value net that compares two positions
            depth -- Max depth to search the tree
            alpha -- The node's alpha value (a numpy board)
            beta -- The node's beta value (a numpy board)
            maximizing_player -- Whether this node is to be max'd or min'd
        Returns:
            The best next position the search came up with (as an NpBoard)
            The best next move the search came up with
        """

        if depth == 0 or gn_current.board(_cache=True).is_game_over():
            # Build the board array and return
            return NpBoard(gn_current.board()), None

        # We need to initialize
        best_move = None

        if maximizing_player:
            # We need to initialize
            best_board = LOSS

            for move in gn_current.board(_cache=True).legal_moves:
                # Make the child
                gn_child = chess.pgn.GameNode()
                gn_child.parent = gn_current
                gn_child.move = move
                # Calculate node position
                child_np_board, _ = Computer.alphabeta(gn_child, comparator, depth-1, alpha, beta, False)
                # See if its better than the current max
                if comparator.greater(child_np_board, best_board):
                    best_board = child_np_board
                    best_move = move

                    # See if we need to change the alpha
                    if comparator.greater(best_board, alpha):
                        alpha = best_board

                        # Check for beta pruning
                        if comparator.less_equal(beta, alpha):
                            break

        else:
            # We need to initialize
            best_board = WIN

            for move in gn_current.board(_cache=True).legal_moves:
                # Make the child
                gn_child = chess.pgn.GameNode()
                gn_child.parent = gn_current
                gn_child.move = move
                # Calculate node position
                child_np_board, _ = Computer.alphabeta(gn_child, comparator, depth-1, alpha, beta, True)
                if comparator.less(child_np_board, best_board):
                    best_board = child_np_board
                    best_move = move

                    # See if we need to change the beta
                    if comparator.less(best_board, beta):
                        beta = best_board

                        # Check for alpha pruning
                        if comparator.less_equal(beta, alpha):
                            break

        return best_board, best_move


class Human(Player):
    def move(self, gn_current):
        bb = gn_current.board()

        print("\nHUMAN\n")
        print(str(bb)[::-1])

        def get_move(move_str):
            try:
                move = chess.Move.from_uci(move_str)
            except:
                print('cant parse')
                return False
            if move not in bb.legal_moves:
                print('not a legal move')
                return False
            else:
                return move

        while True:
            print('your turn:')
            move = get_move(input())
            if move:
                break

        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move

        return gn_new

#
# class Sunfish(Player):
#     def __init__(self, secs=1):
#         self._searcher = sunfish.Searcher()
#         self._pos = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
#         self._secs = secs
#
#     def move(self, gn_current):
#         import sunfish
#
#         assert(gn_current.board().turn == False)
#
#         # Apply last_move
#         crdn = str(gn_current.move)
#         move = (sunfish.parse(crdn[0:2]), sunfish.parse(crdn[2:4]))
#         self._pos = self._pos.move(move)
#
#         t0 = time.time()
#         move, score = self._searcher.search(self._pos, self._secs)
#         print("Time %s : Move %s : Score %s" % (time.time() - t0, move, score))
#         self._pos = self._pos.move(move)
#
#         crdn = sunfish.render(119-move[0]) + sunfish.render(119 - move[1])
#         move = create_move(gn_current.board(), crdn)
#
#         gn_new = chess.pgn.GameNode()
#         gn_new.parent = gn_current
#         gn_new.move = move
#
#         return gn_new
