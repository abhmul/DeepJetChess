import os
import time
import chess
import chess.pgn
import numpy as np
import sunfish
import heapq
import traceback
import random
from datetime import date

from keras.models import load_model

import models

from np_board_utils import sb2array, WIN, LOSS
from comparators import DeepJetChess
from players import Computer, Human, Sunfish

model_func = models.conv10layer
sd = os.getcwd()
wd = os.path.join(sd, '..')
od = os.path.join(wd, 'models')

years = [str(y) for y in [2013, 2014, 2015, 2016]]
model_file = os.path.join(od, '{name}_{years}_weights.h5').format(name=model_func.__name__,
                                                                  years="_".join(years))

MAXD = 1
TOPK=None
CACHE_SIZE=20000

def dump_game(gn_final, side, write_file):

    gn_final.headers["Event"] = "DeepJetChess VS Sunfish"
    gn_final.headers["Site"] = "Abhijeet's Laptop"
    gn_final.headers["Date"] = date.today().strftime("%Y-%m-%d")
    gn_final.headers["White"] = "DeepJetChess"
    gn_final.headers["Black"] = "Sunfish"
    if side == 'A':
        gn_final.headers["Result"] = "1-0"
    elif side == 'B':
        gn_final.headers["Result"] = "0-1"
    else:
        gn_final.headers["Result"] = "1/2-1/2"
    print(gn_final, file=write_file, end="\n\n")

    return True

def game(player_a, player_b, write_game=True):
    # Initialize the game
    gn_current = chess.pgn.Game()

    times = {'A': 0.0, 'B': 0.0}
    # Play until someone loses
    while True:
        # Alternate players
        for side, player in [('A', player_a), ('B', player_b)]:
            # Keep track of how long each player takes per move
            t0 = time.time()
            try:
                gn_current = player.move(gn_current)
            except KeyboardInterrupt:
                return
            except:
                traceback.print_exc()
                return side + '-exception', times

            times[side] += time.time() - t0
            # Log the board change
            print('=========== Player %s: %s' % (side, gn_current.move))
            s = str(gn_current.board(_cache=True))
            print(s)
            if gn_current.board(_cache=True).is_checkmate():
                if write_game:
                    dump_game(gn_current, side, "games.pgn")
                return side, times
            elif gn_current.board().is_stalemate():
                if write_game:
                    dump_game(gn_current, '-', "games.pgn")
                return '-', times
            elif gn_current.board().can_claim_fifty_moves():
                if write_game:
                    dump_game(gn_current, '-', "games.pgn")
                return '-', times
            elif s.find('K') == -1 or s.find('k') == -1:
                # Both AI's suck at checkmating, so also detect capturing the king
                if write_game:
                    dump_game(gn_current, side, "games.pgn")
                return side, times

def play():
    print("Loading the model")
    embedder, comparer = model_func(optimizer=None, mode='play')
    comparator = DeepJetChess(embedder, comparer, cache_size=CACHE_SIZE)
    comparator.load_weights(model_file)
    player_a = Computer(comparator, maxd=MAXD)
    print("Model Loaded!")

    maxd = MAXD
    while True:
        # secs = random.random() * 4
        secs=1
        player_b = Sunfish(secs=secs)
        print("Starting Game")
        side, times = game(player_a, player_b)
        f = open('../stats.txt', 'a')
        f.write('%s %f %f\n' % (side, times['A'], times['B']))
        f.close()

if __name__ == "__main__":
    play()
