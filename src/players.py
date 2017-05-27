import time
import numpy as np

import chess
import chess.pgn
import sunfish

from np_board_utils import sb2array, NpBoard, WIN, LOSS

MAXD = 1
TOPK = 5

def make_gn_child(gn_current, move):
    gn_child = chess.pgn.GameNode()
    gn_child.parent = gn_current
    gn_child.move = move
    return gn_child

def create_move(board, crdn):
    # workaround for pawn promotions
    move = chess.Move.from_uci(crdn)
    if board.piece_at(move.from_square).piece_type == chess.PAWN:
        if int(move.to_square/8) in [0, 7]:
            move.promotion = chess.QUEEN # always promote to queen
    return move

class Player(object):
    def move(self, gn_current):
        raise NotImplementedError()


class Computer(Player):
    def __init__(self, comparator, maxd=MAXD, topk=TOPK, sort=False):
        self._comparator = comparator
        self._gn = None
        self._maxd = maxd
        self._topk = topk
        self._sort = sort
        assert(self._sort or self._topk is not None)

    @property
    def _cache(self):
        return self._comparator._cache

    def move(self, gn_current):
        # assert(gn_current.board(_cache=True).turn == True)

        self._gn = gn_current

        alpha = LOSS
        beta = WIN

        depth = self._maxd
        t0 = time.time()
        # Board in array format
        # Move in python-chess format
        best_board, best_move = self.alphabeta(self._gn, depth, LOSS, WIN, self._gn.board(_cache=True).turn)
        print("Depth %s : Move %s : Time %s : Cache Size %s" %(depth, best_move, time.time() - t0, len(self._cache)))

        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = best_move

        return gn_new

    def alphabeta(self, gn_current, depth, alpha, beta, maximizing_player):
        """
        Runs position alpha-beta pruning using comparator to compare positions
        Arguments:
            gn_current -- The current game node (pgn python-chess format)
            depth -- Max depth to search the tree
            alpha -- The node's alpha value (a numpy board)
            beta -- The node's beta value (a numpy board)
            maximizing_player -- Whether this node is to be max'd or min'd
        Returns:
            The best next position the search came up with (as an NpBoard)
            The best next move the search came up with
        """

        if depth == 0:
            # Build the board array and return
            return NpBoard(gn_current.board(), self._comparator), None
        elif gn_current.board(_cache=True).is_game_over():
            res = gn_current.board().result()
            if res == "1-0":
                return WIN, None if maximizing_player else LOSS, None
            elif res == "0-1":
                return LOSS, None if maximizing_player else WIN, None
            else:
                print("Result of leaf node is not win or loss: %s" % res)
                return NpBoard(gn_current.board(), self._comparator), None

        # We need to initialize
        best_move = None

        # Get all the children
        children = [make_gn_child(gn_current, move) for move in gn_current.board(_cache=True).legal_moves]

        # Cache their embeddings
        np_children = [NpBoard(gn_child.board(), self._comparator) for gn_child in children]
        self._comparator.embed(np_children)

        if maximizing_player:
            # We need to initialize
            best_board = LOSS

            if self._sort:
                child_inds = sorted(range(len(np_children)), key=lambda i: np_children[i], reverse=True)
            else:
                # Get the topk child boards inds
                child_inds = np.argpartition(np_children, -min(self._topk, len(children)))[::-1]
            if self._topk is not None:
                # Filter out the best k moves
                child_inds = child_inds[:self._topk]

            for i in child_inds:
                # Make the child
                gn_child = children[i]
                # Calculate node position
                child_np_board, _ = self.alphabeta(gn_child, depth-1, alpha, beta, False)
                # See if its better than the current max
                if best_board < child_np_board:
                    best_board = child_np_board
                    best_move = gn_child.move

                    # See if we need to change the alpha
                    if alpha < best_board:
                        alpha = best_board

                        # Check for beta pruning
                        if beta <= alpha:
                            break

        else:
            # We need to initialize
            best_board = WIN

            # Sort the child boards inds
            if self._sort:
                child_inds = sorted(range(len(np_children)), key=lambda i: np_children[i])
            else:
                # Get the topk child boards inds
                child_inds = np.argpartition(np_children, min(self._topk, len(children)-1))
            if self._topk is not None:
                # Filter out the best k moves
                child_inds = child_inds[:self._topk]

            for i in child_inds:
                # Make the child
                gn_child = children[i]
                # Calculate node position
                child_np_board, _ = self.alphabeta(gn_child, depth-1, alpha, beta, True)
                if best_board > child_np_board:
                    best_board = child_np_board
                    best_move = gn_child.move

                    # See if we need to change the beta
                    if beta > best_board:
                        beta = best_board

                        # Check for alpha pruning
                        if beta <= alpha:
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


class Sunfish(Player):
    def __init__(self, secs=1):
        self._searcher = sunfish.Searcher()
        self._pos = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
        self._secs = secs

    def move(self, gn_current):
        import sunfish

        assert(gn_current.board().turn == False)

        # Apply last_move
        crdn = str(gn_current.move)
        move = (sunfish.parse(crdn[0:2]), sunfish.parse(crdn[2:4]))
        self._pos = self._pos.move(move)

        t0 = time.time()
        move, score = self._searcher.search(self._pos, self._secs)
        print("Time %s : Avg Secs %s : Move %s : Score %s" % (time.time() - t0, self._secs, move, score))
        self._pos = self._pos.move(move)

        crdn = sunfish.render(119-move[0]) + sunfish.render(119 - move[1])
        move = create_move(gn_current.board(), crdn)

        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move

        return gn_new
