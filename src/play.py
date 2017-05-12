import os
import time
import chess
import chess.pgn
import numpy as np
import sunfish
import heapq
import traceback

from keras.models import load_model

import models

from np_board_utils import sb2array, WIN, LOSS
from comparators import DeepJetChess
from players import Computer, Human

model_func = models.incpetion_net
sd = os.getcwd()
wd = os.path.join(sd, '..')
od = os.path.join(wd, 'models')

years = [str(y) for y in [2013, 2014, 2015, 2016]]
model_file = os.path.join(od, '{name}_{years}_weights.h5').format(name=model_func.__name__,
                                                                  years="_".join(years))

MAXD = 3

def game(player_a, player_b):
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
                return side, times
            elif gn_current.board().is_stalemate():
                return '-', times
            elif gn_current.board().can_claim_fifty_moves():
                return '-', times
            elif s.find('K') == -1 or s.find('k') == -1:
                # Both AI's suck at checkmating, so also detect capturing the king
                return side, times

def play():
    print("Loading the model")
    model = model_func()
    model.load_weights(model_file)
    print("Model Loaded!")
    # model.save(model_file + ".compiled")
    # model = load_model(model_file + ".compiled")

    maxd = MAXD
    while True:
        player_a = Computer(DeepJetChess(model), maxd=MAXD)
        player_b = Human()
        print("Starting Game")
        side, times = game(player_a, player_b)
        f = open('../stats.txt', 'a')
        f.write('%s %f %f\n' % (side, times['A'], times['B']))
        f.close()

if __name__ == "__main__":
    play()
