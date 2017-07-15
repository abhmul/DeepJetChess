import os

import torch.optim as optim
from torch.autograd import Variable

from chess_precompute_dataset import ChessDataset
from torch_models import chess_loss, accuracy, chess_loss2, avg_val, avg_prob
from torch_models import SimpleModel2 as model_class
import torch_models as tm
from pyjet.data import DatasetGenerator
import pyjet.backend as J
from pyjet.callbacks import Plotter, ModelCheckpoint
from pyjet.training import GeneratorEnqueuer

# Define the globals
BATCH_SIZE = 32
EPOCHS = 200

SOURCE = os.getcwd()
WORKING = os.path.join(SOURCE, '..')
OUTPUT = os.path.join(WORKING, 'models')

FN_IN = os.path.join(WORKING, 'grandmaster_games.h5')

MODEL_FILE = os.path.join(OUTPUT, 'test_' + model_class.name + '_epoch{epoch}_weights.state')
# SAVE_FILE =  MODEL_FILE.format(name="alpha_chess2",
                            #    years="_".join(TRAIN_YEARS + VAL_YEARS))

# Load the train and val chess dataset
full_dataset = ChessDataset(FN_IN)
train_dataset, val_dataset = full_dataset.train_val_split(split=0.2, seed=1234)


# Print some dataset metrics
print("Number of Train Games: ", len(train_dataset))
print("Number of Val Games: ", len(val_dataset))

# Instantiate the generators
traingen = GeneratorEnqueuer(DatasetGenerator(train_dataset, batch_size=BATCH_SIZE, shuffle=True, seed=1234))
valgen = GeneratorEnqueuer(DatasetGenerator(val_dataset, batch_size=BATCH_SIZE, shuffle=True, seed=1234))

try:
    # Start the parallel queue generation
    traingen.start(max_q_size=3)
    valgen.start(max_q_size=3)

    # Setup the model
    alpha_chess = model_class()

    # Setup the optimizer
    sgd = optim.Adam(alpha_chess.parameters())

    # Setup the loss
    loss = chess_loss2

    # This will save the best scoring model weights to the parent directory
    best_model = ModelCheckpoint(MODEL_FILE, monitor='accuracy', monitor_val=True, mode='max',
                                 verbose=1, save_best_only=False)
    plotter = Plotter(scale='log', monitor=loss.__name__)
    callbacks = [best_model, plotter]

    alpha_chess.fit_generator(traingen, EPOCHS, traingen._generator.steps_per_epoch, sgd, chess_loss2,
                              validation_generator=valgen, validation_steps=valgen._generator.steps_per_epoch,
                              metrics=[accuracy, avg_val, avg_prob, tm.avg_min_prob, tm.avg_min_val], callbacks=callbacks)
finally:
    traingen.stop()
    valgen.stop()
