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
from tqdm import tqdm

try:
   import cPickle as pickle
except:
   import pickle

from torch_models import SimpleModel2 as model_class

from np_board_utils import sb2array, WIN, LOSS
from players import Sunfish, Net



SOURCE = os.getcwd()
WORKING = os.path.join(SOURCE, '..')
OUTPUT = os.path.join(WORKING, 'models')

MODEL_FILE = os.path.join(OUTPUT, "alpha_chess_2010_2011_2013_2014_2015_2016_weights" + ".state")


def dump_game(gn_final, side, write_filename):

    # record the Result
    gn_root = gn_final.root()
    if side == 'A':
        gn_root.headers["Result"] = "1-0"
    elif side == 'B':
        gn_root.headers["Result"] = "0-1"
    else:
        gn_root.headers["Result"] = "1/2-1/2"

    with open(write_filename, 'a') as write_file:
        print(gn_final, file=write_file, end="\n\n")

    return True

def game(player_a, player_b, write_game=True):
    # Initialize the game
    gn_root = chess.pgn.Game()

    # Record the stats
    gn_root.headers["Event"] = "AlphaChess VS Sunfish"
    gn_root.headers["Site"] = "Abhijeet's Laptop"
    gn_root.headers["Date"] = date.today().strftime("%Y-%m-%d")
    gn_root.headers["White"] = "AlphaChess"
    gn_root.headers["Black"] = "Sunfish"

    gn_current = gn_root

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
            # print('=========== Player %s: %s' % (side, gn_current.move))
            # s = str(gn_current.board(_cache=True))
            # print(s)
            if gn_current.board().is_game_over():
                return gn_current.board().result(), times

def play(model_num, games=200):
    # model_name = "alpha_chess2_epoch{}_weights.state".format(model_num)
    model_name = "test_{}_epoch{}_weights.state".format(model_class.name, model_num)
    print("Loading the model: ", model_name)
    alpha_chess = model_class()
    player_a = Net(alpha_chess, name="AlphaChess")
    player_a.load_state(os.path.join(OUTPUT, model_name))
    alpha_chess2 = model_class()
    reward_sum = 0
    net_times = 0
    sunfish_times = 0
    # player_b = Net(alpha_chess2, name="AlphaChess")
    # player_b.load_state(os.path.join(OUTPUT, "alpha_chess_2010_2011_2013_2014_2015_2016_weights" + ".state"))
    print("Model Loaded!")
    for i in tqdm(range(games)):
        secs = random.random() * .001
        # secs=.1
        player_b = Sunfish(secs=secs)
        # print("Starting Game")
        reward = 0
        if i % 2 == 0:
            result, times = game(player_a, player_b)
            if result == "1-0":
                reward = 1
            elif result == "0-1":
                reward = -1
            # print("%s : %s  %f : %s %f" % (reward, "N", times['A'], "S", times['B']))
            net_times += times['A']
            sunfish_times += times['B']
        else:
            result, times = game(player_b, player_a)
            if result == "1-0":
                reward = -1
            elif result == "0-1":
                reward = 1
            # print("%s : %s  %f : %s %f" % (reward, "N", times['B'], "S", times['A']))
            net_times += times['B']
            sunfish_times += times['A']
        reward_sum += reward
    print("\tAverage Reward for %s: %f" % (model_num, reward_sum / games))
    print("\tAverage Net Time for %s: %f" % (model_num, net_times / games))
    print("\tAverage Sunfish Time for %s: %f" % (model_num, sunfish_times / games))
    return reward_sum / games
        # if result == "1/2-1/2":
        #     winner = "-"
        # f = open('../stats{}.txt'.format(model_num), 'a')
        # f.write('%s %f %f\n' % (winner, times['A'], times['B']))
        # f.close()

if __name__ == "__main__":
    xs = []
    ys = []
    for i in range(50, -1, -1):
        xs.append(i)
        ys.append(play(i, games=50))

    import matplotlib.pyplot as plt
    plt.plot(xs, ys)
    # embedder, comparer = model_func(optimizer=None, mode='play')
    # comparator = DeepJetChess(embedder, comparer, cache_size=CACHE_SIZE)
    # comparator.load_weights(model_file)

    # for depths in range(1, 6):
        # player = Computer(comparator, maxd=depths, sort=SORT, topk=TOPK)

        # secs=1
        # player = Sunfish(secs=secs)
        # elo_test(player)
