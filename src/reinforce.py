import os
import time
import random
import numpy as np
import chess
import chess.pgn
import heapq
import traceback
from datetime import date
from copy import deepcopy
import multiprocessing

from torch_models import chess_loss
from torch_models import SimpleModel2 as model_class
import torch.optim as optim

from np_board_utils import NpBoard
from players import Learner, Opponent

class PrevNetPool(object):

    def __init__(self, base_net, save_id=0, save_dir="../models/"):
        self.pool = []
        self.save_dir = save_dir
        self.save_id = save_id
        self.base_net = base_net

    def __len__(self):
        return len(self.pool)

    def sample(self):
        load_path = random.choice(self.pool)
        self.base_net.load_state(load_path)
        print("Sampled %s" % load_path)
        return self.base_net

    def insert(self, net):
        save_path = os.path.join(self.save_dir,"PrevNet" + str(self.save_id) + ".state")
        new_net = deepcopy(net.net)
        new_net.save_state(save_path)
        self.pool.append(save_path)
        self.save_id += 1

def game(learner, opponent, play_white=True):
    # Play the game
    learner_moves = []
    learner_preds = []
    # Initialize the game
    gn_root = chess.pgn.Game()
    gn = gn_root
    # Set up white and black
    white = learner if play_white else opponent
    black = opponent if play_white else learner
    current, other = white, black
    # print(current.name)
    # print(other.name)
    # Run the entire game until finish
    while True:
        gn, move_ind, torch_preds = current.move(gn)
        # If learner is playing and the game isn't over

        if torch_preds is not None and current.name == learner.name:
            learner_moves.append(move_ind)
            learner_preds.append(torch_preds)
        if gn.board(_cache=True).is_game_over():
            break
        # Switch who's turn it is
        current, other = other, current

    return gn.board().result(), learner_moves, learner_preds



def run_n_games(learner, opponent, num_games):

    reward_sum = 0
    game_i = 0
    while game_i < num_games:
        # Figure out if we're white
        play_white = (game_i % 2 == 1)
        # Play the game
        result, learner_moves, learner_preds = game(learner, opponent, play_white)
        # Throw away drawn games
        # print(result)
        if result not in {"1-0", "0-1"}:
            continue
        # Calculate the reward
        reward = 1 if result == "1-0" else -1
        reward *= 1 if play_white else -1
        # print(reward)
        # Create the target Tensor
        learner_moves = np.asarray(learner_moves, dtype=int)
        assert(len(learner_preds) == len(learner_moves))
        # print(learner_moves)
        # print(learner.cast_target_to_torch(learner_moves))
        # Calculate the game loss
        loss = reward * chess_loss(learner_preds, learner.cast_target_to_torch(learner_moves))
        # Backpropogate the loss
        loss.backward()
        # Increment the number of games played
        game_i += 1
        # Accumulate the rewards
        reward_sum += reward
    return float(reward_sum) / num_games

def run_n_games_parallel(learner, opponent, num_games, pool):

    reward_sum = 0
    game_i = 0
    # print(num_games)
    while game_i < num_games:
        games_run = 0
        game_args = [(learner, opponent, i % 2 == 1) for i in range(game_i, num_games)]
        for result, learner_moves, learner_preds in pool.starmap(game, game_args):
            # Throw away drawn games
            print(result)
            if result not in {"1-0", "0-1"}:
                continue
            # Calculate the reward
            reward = 1 if result == "1-0" else -1
            reward *= 1 if play_white else -1
            # print(reward)
            # Create the target Tensor
            learner_moves = np.asarray(learner_moves, dtype=int)
            assert(len(learner_preds) == len(learner_moves))
            # print(learner_moves)
            # print(learner.cast_target_to_torch(learner_moves))
            # Calculate the game loss
            loss = reward * chess_loss(learner_preds, learner.cast_target_to_torch(learner_moves))
            # Backpropogate the loss
            loss.backward()
            # Increment the number of games played
            games_run += 1
            # Accumulate the rewards
            reward_sum += reward
        game_i += games_run
    return float(reward_sum) / game_i

class ReinforcementLearner(object):

    def __init__(self, initial_net, optimizer, batch_size=64, pool=None):
        self.prev_net_pool = PrevNetPool(Opponent(deepcopy(initial_net.net), name="PrevNet"))
        self.curr_net = initial_net
        self.batch_counter = 0
        self.rewards = []
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.pool = pool

    def log_stats(self):
        print("Average Reward: %s" % np.average(self.rewards))


    def start(self, save_interval=500):

        while True:
            if self.batch_counter % save_interval == 0:
                print("Saving Current Net")
                self.prev_net_pool.insert(self.curr_net)
                if self.batch_counter != 0:
                    self.log_stats()
                    self.rewards = []

            opponent = self.prev_net_pool.sample()
            # opponent = self.prev_net_pool.base_net
            print("Curr Net: ", self.curr_net.name)
            self.optimizer.zero_grad()
            if self.pool is None:
                batch_reward = run_n_games(self.curr_net, opponent, self.batch_size)
            else:
                batch_reward = run_n_games_parallel(self.curr_net, opponent, self.batch_size, self.pool)
            print("Batch %s Reward: %s" % (self.batch_counter, batch_reward))
            self.optimizer.step()

            self.rewards.append(batch_reward)
            self.batch_counter += 1


if __name__ == "__main__":
    SOURCE = os.getcwd()
    WORKING = os.path.join(SOURCE, '..')
    OUTPUT = os.path.join(WORKING, 'models')

    print("Loading the model")
    alpha_chess = model_class()
    initial_net = Learner(alpha_chess, name="Learner")
    initial_net.load_state(os.path.join(OUTPUT, "test_{}_epoch43_weights.state".format(model_class.name)))
    initial_net.name = "Learner"

    sgd = optim.SGD(alpha_chess.parameters(), lr=0.001)

    # pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    # print("Running on %s processors" % (multiprocessing.cpu_count() - 1))
    pool = None

    # Start the learning
    rl = ReinforcementLearner(initial_net, sgd, pool=pool)
    rl.start()
