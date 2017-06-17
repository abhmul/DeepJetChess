import os

import torch.optim as optim
from torch.autograd import Variable

from chess_dataset import ChessDataset
from torch_models import AlphaChess, chess_loss, accuracy
from pyjet.data import DatasetGenerator
import pyjet.backend as J
from pyjet.callbacks import Plotter, ModelCheckpoint
from pyjet.training import GeneratorEnqueuer

# Define the globals
BATCH_SIZE = 32
EPOCHS = 200

TRAIN_YEARS = [str(y) for y in [2010, 2011, 2013, 2014, 2015]]
VAL_YEARS = [str(y) for y in [2016]]

SOURCE = os.getcwd()
WORKING = os.path.join(SOURCE, '..')
OUTPUT = os.path.join(WORKING, 'models')

FN_IN = os.path.join(WORKING, 'cvc_{year}_more.h5')
TRAIN_FN_INS = [FN_IN.format(year=year) for year in TRAIN_YEARS]
VAL_FN_INS = [FN_IN.format(year=year) for year in VAL_YEARS]

MODEL_FILE = os.path.join(OUTPUT, 'alpha_chess2_epoch{epoch}_weights.state')
# SAVE_FILE =  MODEL_FILE.format(name="alpha_chess2",
                            #    years="_".join(TRAIN_YEARS + VAL_YEARS))

# Load the train and val chess dataset
train_dataset = ChessDataset()
train_dataset.load_many_h5(TRAIN_FN_INS)
val_dataset = ChessDataset()
val_dataset.load_many_h5(VAL_FN_INS)

# Print some dataset metrics
print("Number of Train Games: ", len(train_dataset))
print("Number of Val Games: ", len(val_dataset))

# Instantiate the generators
traingen = GeneratorEnqueuer(DatasetGenerator(train_dataset, batch_size=BATCH_SIZE, shuffle=True, seed=1234))
valgen = GeneratorEnqueuer(DatasetGenerator(val_dataset, batch_size=BATCH_SIZE, shuffle=True, seed=1234))
try:
    # Start the parallel queue generation
    traingen.start()
    valgen.start()

    # Setup the model
    alpha_chess = AlphaChess()

    # Setup the optimizer
    sgd = optim.Adam(alpha_chess.parameters())

    # This will save the best scoring model weights to the parent directory
    best_model = ModelCheckpoint(MODEL_FILE, monitor='accuracy', monitor_val=True, mode='max',
                                 verbose=1, save_best_only=False)
    plotter = Plotter(scale='log', monitor='chess_loss')
    callbacks = [best_model, plotter]

    alpha_chess.fit_generator(traingen, EPOCHS, traingen._generator.steps_per_epoch, sgd, chess_loss,
                              validation_generator=valgen, validation_steps=valgen._generator.steps_per_epoch,
                              metrics=[accuracy], callbacks=callbacks)
finally:
    traingen.stop()
    valgen.stop()
