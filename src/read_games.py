import os
from collections import namedtuple
import argparse
import time

import numpy as np
import chess
import chess.pgn
import h5py
from multiprocessing import Process, Lock, Pool, cpu_count
from threading import Semaphore # This is needed to fix imap memory leak
# https://stackoverflow.com/questions/40922526/memory-usage-steadily-growing-for-multiprocessing-pool-imap-unordered
from timeit import timeit
from functools import partial
import logging

from np_board_utils import sb2array, array2b, make_gn_child, split_boards, switch_state_sides, switch_board_sides, square2ind

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s:%(process)d:%(thread)d:%(message)s')
logger = logging.getLogger()

import sys
sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser(description='Read and merge pgn games into an h5 file.')
parser.add_argument('h5_fname_out',
                   help='the h5 fname to parse the games into')
parser.add_argument('pgn_fname_ins', type=str, nargs='+',
                   help='The pgns to parse')
parser.add_argument('-i', '--hsize', type=int, default=1e6,
                   help='The size of the shared buffer')
parser.add_argument('-p', '--processes', type=int, default=0,
                   help='The number of processes to run on')
parser.add_argument('-l', '--log_interval', type=int, default=1000,
                   help='Will cause program to log every n games')
parser.add_argument('-c', '--chunksize', type=int, default=100,
                   help='Chunksize with which to parse games in parallel.')
parser.add_argument('-m', '--maxtasksperchild', type=int, default=-1,
                   help='Maximum number of tasks a child can run before it is replaced.')

RESULTS = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}
State = namedtuple('State', ['legal_boards', 'castling', 'en_passants'])
# Game = namedtuple('Game', ['states', 'turns', 'winner', 'move_made_indicies'])
Game = namedtuple('Game', ['states', 'winner', 'move_made_indicies'])

def build_game(game):
    """Returns a list of all the boards in order in a game"""
    gn = game.end()
    game_nodes = []
    # Loop through the game boards from end to start
    while gn:
        gn.board(_cache=True)
        game_nodes.append(gn)
        gn = gn.parent
    # Reverse so they're start to end
    game_nodes.reverse()
    return game_nodes

def create_move_arr(game_nodes):
    # Create the move array and set the last move to 0
    move_arr = np.array([1 if gn.board().turn else -1 for gn in game_nodes], dtype=np.int8)
    move_arr[-1] = 0
    return move_arr

def create_castle_arr(board, out=None):
    if out is None:
        out = np.zeros((4,), dtype=np.int8)
    else:
        assert(out.shape == (4,))

    if board.has_kingside_castling_rights(chess.WHITE):
        out[0] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        out[1] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        out[2] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        out[3] = 1

    return out

def process_game(game_root):
    # logging.info("Producing")
    game_result = game_root.headers['Result']
    # If we have a normal result,
    if game_result in RESULTS:
        winner = RESULTS[game_result]
        if not winner:
            # Throw out draws
            return None
    else:
        logging.info('Game had invalid result %s, continuing to next game' % game_result)
        return None

    game_nodes = build_game(game_root)
    if len(game_nodes) < 4:
        # Game is too short, skip it
        return None

    # move_arr = create_move_arr(game_nodes)[:-1]
    assert(winner != 0)
    # win_arr = np.full((len(game_nodes),), winner, dtype=np.int8)
    legal_states = []
    move_made_indicies = []
    # Iterate over each node and create all the legal moves, and leave off the end
    # Include the next move so we can figure out which move was made
    # Skip moves made by the loser
    start = 0 if winner == 1 else 1
    for gn, gn_next in zip(game_nodes[start:-1:2], game_nodes[start+1::2]):
        # Check that we only write the boards of the winner
        assert(gn.board().turn == chess.WHITE if winner == 1 else gn.board().turn == chess.BLACK)
        gn_children = [make_gn_child(gn, move) for move in gn.board().legal_moves]
        # Initialize the output arrays
        np_legal_move_boards = np.empty((len(gn_children), 8, 8), dtype=np.int8)
        legal_c = np.zeros((len(gn_children), 4), dtype=np.int8)
        legal_ep = np.zeros((len(gn_children), 2), dtype=np.int8)
        move_made_index = -1
        # Turn the board states into numpy states
        for i, legal_gn in enumerate(gn_children):
            board = legal_gn.board()
            np_legal_move_boards[i] = sb2array(str(board))
            # Stores castling rights in the order WK WQ BK BQ
            create_castle_arr(board, out=legal_c[i])
            # Store the en passant square
            if board.has_legal_en_passant():
                legal_ep[i, 0], legal_ep[i, 1] = square2ind(board.ep_square)
            # See if this was the move made
            if gn_next.move == legal_gn.move:
                # It should not have already been set to something else
                assert(move_made_index == -1)
                move_made_index = i
        # We should have assigned a move
        assert(move_made_index != -1)
        state = State(np_legal_move_boards, legal_c, legal_ep)
        # Switch the state side if black won
        if winner == -1:
            state = State(*switch_state_sides(state.legal_boards, state.castling, state.en_passants))
        legal_states.append(state)
        move_made_indicies.append(move_made_index)

    # assert(len(move_arr) == len(legal_states) == len(move_made_indicies))
    assert(len(legal_states) == len(move_made_indicies))
    # return Game(legal_states, move_arr, int(winner == 1), np.asarray(move_made_indicies, dtype=np.int8))
    return Game(legal_states, int(winner == 1), np.asarray(move_made_indicies, dtype=np.uint8))

def parallel_parse(parser, pool, chunksize=100):
    for game in pool.imap_unordered(process_game, parser, chunksize=chunksize):
        # sem.release()
        if game is None:
            continue
        yield game

def sequential_parse(parser, pool=None, chunksize=None):
    assert(pool is None)
    for game in map(process_game, parser):
        # sem.release()
        if game is None:
            continue
        yield game

class PgnParser(object):

    def __init__(self, pgn_fname):
        self.pgn_fname = pgn_fname
        logging.info("Opening %s" % pgn_fname)
        self.f = None
        self.f_size = os.path.getsize(self.pgn_fname)
        # self.sem = semaphore

    def __iter__(self):
        self.open_pgn()
        return self


    def __next__(self):
        game_root = None
        assert(self.f is not None)
        while True:
            try:
                # Critical Section: we need the lock when reading the file (not sure if thread safe)
                # self.lock.acquire()
                game_root = chess.pgn.read_game(self.f)
            except KeyboardInterrupt:
                self.close_pgn()
                raise
            except:
                logging.info("Failed to read game, contniuing to next one.")
                continue
            finally:
                # self.lock.release()
                pass

            if not game_root:
                self.close_pgn()
                raise StopIteration

            # self.sem.acquire()
            return game_root

    def __length_hint__(self):
        return self.f_size

    def open_pgn(self):
        self.f = open(self.pgn_fname)

    def close_pgn(self):
        self.f.close()
        self.f = None

def parse_all_games(fname_out, parsers, pool=None, init_size=1e6, chunksize=100, log_interval=1000):

    parse = parallel_parse if pool is not None else sequential_parse

    # Estimated size is 91.6 GB
    board_size = init_size # 40 * 35 * init_size # 35 is the avg number of legal moves per state
    state_size = init_size # 40 * init_size # 40 is the avg number of moves per game
    game_size = init_size

    logging.info("Creating Dataset...")
    g = h5py.File(fname_out, 'w')
    X = g.create_dataset('X', (board_size, 8, 8), dtype='i8', maxshape=(None, 8, 8), chunks=True, compression="lzf")
    # M = g.create_dataset('M', (state_size,), dtype='i8', maxshape=(None,), chunks=True)
    MM = g.create_dataset('MM', (state_size,), dtype='u8', maxshape=(None,), chunks=True, compression="lzf")
    W = g.create_dataset('W', (game_size,), dtype='i8', maxshape=(None,), chunks=True, compression="lzf")
    C = g.create_dataset('C', (board_size, 4), dtype='i8', maxshape=(None, 4), chunks=True, compression="lzf")
    E = g.create_dataset('E', (board_size, 2), dtype='i8', maxshape=(None, 2), chunks=True, compression="lzf")
    # Index of beginning of state in boards length datasets
    S_ind = g.create_dataset('S_ind', (state_size,), dtype=int, maxshape=(None,), chunks=True, compression="lzf")
    # Index of beginning of game in state length datasets
    G_ind = g.create_dataset('G_ind', (game_size,), dtype=int, maxshape=(None,), chunks=True, compression="lzf")
    logging.info("Dataset Iniitialized!")

    # Set up the offset trackers
    board_num = 0
    state_num = 0
    game_num = 0
    start_time = time.time()
    iter_time = start_time
    # Now fill the dataset
    for parser in parsers:
        for game in parse(parser, pool, chunksize=chunksize):
            # logging.info("Consuming %d" % game_num)
            # Fill in the datasets
            # Resize the arrays if necessary
            while game_num + 1 > game_size:
                game_size *= 2
                logging.info("Resizing Game Size to %s" % game_size)
                [d.resize(size=game_size, axis=0) for d in (W, G_ind)]
            # Fill in Game level info
            W[game_num] = game.winner
            G_ind[game_num] = state_num
            # Resize the arrays if necessary
            while state_num + len(game.states) > state_size:
                state_size *= 2
                logging.info("Resizing State Size to %s" % state_size)
                [d.resize(size=state_size, axis=0) for d in (MM, S_ind)]
            # Fill in state level info
            # M[state_num:state_num + len(game.states)] = game.turns
            MM[state_num:state_num + len(game.states)] = game.move_made_indicies

            for state in game.states:
                # Resize the arrays if necessary
                while board_num + len(state.legal_boards) > board_size:
                    board_size *= 2
                    logging.info("Resizing Board Size to %s" % board_size)
                    [d.resize(size=board_size, axis=0) for d in (X, C, E)]
                # Fill in board level info
                assert(len(state.legal_boards) == len(state.castling) == len(state.en_passants))
                assert(state.legal_boards.shape[1:] == (8, 8))
                assert(state.castling.shape[1:] == (4,))
                assert(state.en_passants.shape[1:] == (2,))
                X[board_num:board_num + len(state.legal_boards)] = state.legal_boards
                C[board_num:board_num + len(state.castling)] = state.castling
                E[board_num:board_num + len(state.en_passants)] = state.en_passants
                S_ind[state_num] = board_num
                board_num += len(state.legal_boards)
                state_num += 1

            game_num += 1
            if game_num % log_interval == 0:
                curr_time = time.time()
                rate = (curr_time - iter_time)
                elapsed = (curr_time - start_time)
                avg_rate = elapsed / game_num
                print("Game Number: ", game_num)
                print("Elapsed Time - %sm : Iteration Time - %sm : Avg Rate - %ss" % (elapsed/60, rate/60, avg_rate))
                print("\tNumber of States: ", state_num)
                print("\tNumber of Boards: ", board_num)
                iter_time = curr_time
                # time.sleep(0.1)

    # Now resize the arrays down to their actual sizes
    logging.info("Final Game Size: %s" % game_num)
    [d.resize(size=game_num, axis=0) for d in (W, G_ind)]
    logging.info("Final State Size %s" % state_num)
    # [d.resize(size=state_num, axis=0) for d in (M, MM, S_ind)]
    [d.resize(size=state_num, axis=0) for d in (MM, S_ind)]
    logging.info("Final Board Size %s" % board_num)
    [d.resize(size=board_num, axis=0) for d in (X, C, E)]
    g.close()

if __name__ == "__main__":
    args = parser.parse_args()
    logging.info("Reading in files: " + " ,".join(args.pgn_fname_ins))
    parsers = [PgnParser(fname_in) for fname_in in args.pgn_fname_ins]

    logging.info("Running on %s processors" % (args.processes+1))
    if args.maxtasksperchild > 0:
        maxtasksperchild = args.maxtasksperchild
        logging.info("Running with %s Max tasks per child" % args.maxtasksperchild)
    else:
        maxtasksperchild = None

    pool = Pool(processes=args.processes, maxtasksperchild=maxtasksperchild) if args.processes > 0 else None

    logging.info("Initializinng H5 size to %s games" % args.hsize)
    if pool is not None:
        logging.info("Using Chunksize of %s" % (args.chunksize))
    logging.info("Setting up to log every %s games" % args.log_interval)
    # Start the consumer
    parse_all_games(args.h5_fname_out, parsers, pool=pool, init_size=args.hsize, chunksize=args.chunksize, log_interval=args.log_interval)
