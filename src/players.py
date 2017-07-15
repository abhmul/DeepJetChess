import os
import time
import numpy as np

import chess
import chess.pgn
import sunfish
import tools

from np_board_utils import sb2array, NpBoard, WIN, LOSS, switch_input_sides, split_boards, square2ind

def make_gn_child(gn_current, move):
    gn_child = chess.pgn.GameNode()
    gn_child.parent = gn_current
    gn_child.move = move
    return gn_child

def create_move(board, crdn):
    # workaround for pawn promotions
    move = chess.Move.from_uci(crdn)
    print(board)
    print(move.from_square)
    print(move.to_square)
    if board.piece_at(move.from_square).piece_type == chess.PAWN:
        if int(move.to_square/8) in [0, 7]:
            move.promotion = chess.QUEEN # always promote to queen
    return move

def create_input(gn_children, flatten=True):
    np_legal_move_boards = np.empty((len(gn_children), 8, 8))
    model_input = np.zeros((len(gn_children), 8, 8, 17))
    for i, gn_child in enumerate(gn_children):
        child_board = gn_child.board()
        np_legal_move_boards[i] = sb2array(str(child_board))
        # Set castling rights
        if child_board.has_kingside_castling_rights(chess.WHITE):
            model_input[i, :, :, 12] = 1
        if child_board.has_queenside_castling_rights(chess.WHITE):
            model_input[i, :, :, 13] = 1
        if child_board.has_kingside_castling_rights(chess.BLACK):
            model_input[i, :, :, 14] = 1
        if child_board.has_queenside_castling_rights(chess.BLACK):
            model_input[i, :, :, 15] = 1
        # Set the en passant square
        if child_board.has_legal_en_passant():
            row, col = square2ind(child_board.ep_square)
            model_input[i, row, col, 16] = 1
    model_input[:, :, :, :12] = split_boards(np_legal_move_boards)
    return model_input

def flatten_model_input(model_input):
    flat_sample = np.zeros((model_input.shape[0], 8 * 8 * 13 + 4))
    flat_sample[:, :8 * 8 * 12] = model_input[:, :12].reshape(model_input.shape[0], -1)
    flat_sample[:, 8*8*12:8*8*13] = model_input[:, 16:].reshape(model_input.shape[0], -1)
    flat_sample[:, -4:] = model_input[:, 12:16, 0, 0]
    return flat_sample

class Player(object):
    def move(self, gn_current):
        raise NotImplementedError()

class Net(Player):

    def __init__(self, net, name="Net"):
        self.net = net
        self.net.cast_model_to_cuda()
        self._gn = None
        self.name = name
        self.__result_map = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}

    def move(self, gn_current):

        self._gn = gn_current
        t0 = time.time()
        # Get all the children
        gn_children = [make_gn_child(gn_current, move) for move in gn_current.board(_cache=True).legal_moves]
        # Check if any of the moves are a win and return if so
        for i, gn_child in enumerate(gn_children):
            if gn_child.board(_cache=True).is_game_over() and gn_child.board().result() in {"0-1", "1-0"}:
                return gn_child
        # Turn them into numpy arrays
        model_input = create_input(gn_children)
        # If the side is black, switch it
        if gn_current.board().turn == chess.BLACK:
            model_input = switch_input_sides(model_input)
        # Turn into pytorch format
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = flatten_model_input(model_input)
        # Score the positions
        prob_dist = self.net.predict_on_batch([model_input])[0]
        # Print the top moves and their probs for logging
        # sorted_prob_dist_inds = np.argsort(prob_dist)[::-1][:5]
        # print("Original Board:")
        # print(gn_current.board())
        # for ind in sorted_prob_dist_inds:
        #     print("p = ", prob_dist[ind])
        #     print(gn_children[ind].board(), '\n')
        # print("\n")

        # print(prob_dist.shape)
        assert(prob_dist.shape == (len(gn_children),))
        # Sample from the distribution
        child_ind = np.random.choice(np.arange(len(gn_children)), p=prob_dist)
        # child_ind = np.argmax(prob_dist)
        # print("Time %s" % (time.time() - t0))
        return gn_children[child_ind]

    def save_state(self, fname):
        self.net.save_state(fname)

    def load_state(self, fname):
        self.name = os.path.splitext(os.path.basename(fname))[0]
        self.net.load_state(fname)

    def cast_target_to_torch(self, y, volatile=False):
        return self.net.cast_target_to_torch(y, volatile=volatile)

class Learner(Net):

    def move(self, gn_current):
        self._gn = gn_current
        t0 = time.time()
        # Get all the children
        gn_children = [make_gn_child(gn_current, move) for move in gn_current.board(_cache=True).legal_moves]
        # Check if any of the moves are a win and return if so
        for i, gn_child in enumerate(gn_children):
            if gn_child.board(_cache=True).is_game_over() and gn_child.board().result() in {"0-1", "1-0"}:
                return gn_child, i, None
        # Turn them into numpy arrays
        model_input = create_input(gn_children)
        # If the side is black, switch it
        if gn_current.board().turn == chess.BLACK:
            model_input = switch_input_sides(model_input)
        # Turn into pytorch format
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = flatten_model_input(model_input)
        # Compute the forward and store the gradients
        torch_input = self.net.cast_input_to_torch([model_input])
        torch_preds = self.net(torch_input)
        # Score the positions
        prob_dist = self.net.cast_output_to_numpy(torch_preds)[0]
        assert(prob_dist.shape == (len(gn_children),))
        assert(torch_preds[0].size(0) == prob_dist.shape[0] and len(torch_preds[0].size()) == 1)
        # Sample from the distribution
        child_ind = np.random.choice(np.arange(len(gn_children)), p=prob_dist)
        return gn_children[child_ind], child_ind, torch_preds[0]

class Opponent(Net):

    def move(self, gn_current):
        self._gn = gn_current
        t0 = time.time()
        # Get all the children
        gn_children = [make_gn_child(gn_current, move) for move in gn_current.board(_cache=True).legal_moves]
        # Check if any of the moves are a win and return if so
        for i, gn_child in enumerate(gn_children):
            if gn_child.board(_cache=True).is_game_over() and gn_child.board().result() in {"0-1", "1-0"}:
                return gn_child, i, None
        # Turn them into numpy arrays
        model_input = create_input(gn_children)
        # If the side is black, switch it
        if gn_current.board().turn == chess.BLACK:
            model_input = switch_input_sides(model_input)
        # Turn into pytorch format
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = flatten_model_input(model_input)
        # Compute the forward and store the gradients
        torch_input = self.net.cast_input_to_torch([model_input], volatile=True)
        torch_preds = self.net(torch_input)
        # Score the positions
        prob_dist = self.net.cast_output_to_numpy(torch_preds)[0]
        assert(prob_dist.shape == (len(gn_children),))
        assert(torch_preds[0].size(0) == prob_dist.shape[0] and len(torch_preds[0].size()) == 1)
        # Sample from the distribution
        child_ind = np.random.choice(np.arange(len(gn_children)), p=prob_dist)
        return gn_children[child_ind], child_ind,None

class Sunfish(Player):
    def __init__(self, secs=1, initial_pos = None):
        self._searcher = sunfish.Searcher()
        if initial_pos is not None:
            self._pos = tools.parseFEN(initial_pos.board().fen())
        else:
            self._pos = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
        self._secs = secs

    def move(self, gn_current):

        # assert(gn_current.board().turn ==)

        # Apply last_move
        # crdn = str(gn_current.move)
        # move = (sunfish.parse(crdn[0:2]), sunfish.parse(crdn[2:4]))
        # self._pos = self._pos.move(move)

        self._pos = tools.parseFEN(gn_current.board().fen())

        t0 = time.time()
        move, score = self._searcher.search(self._pos, self._secs)
        san = tools.renderSAN(self._pos, move)
        # print("Time %s : Avg Secs %s : Move %s : Score %s" % (time.time() - t0, self._secs, san, score))
        self._pos = self._pos.move(move)

        # crdn = sunfish.render(119-move[0]) + sunfish.render(119 - move[1])
        # move = create_move(gn_current.board(), crdn)
        move = gn_current.board().parse_san(san)

        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move

        return gn_new
