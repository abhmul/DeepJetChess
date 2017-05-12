import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD, RMSprop, Adam
from dataprocessor5 import DataProcessor, VERBOSITY
import models

from plotter_callback import Plotter

BATCH_SIZE = 32
EPOCH_STEPS = int((1000000 + BATCH_SIZE - 1) / BATCH_SIZE)
# EPOCH_STEPS = 1
VAL_STEPS = int(0.15 * EPOCH_STEPS) + 1


sd = os.getcwd()
wd = os.path.join(sd, '..')
od = os.path.join(wd, 'models')

model_func = models.conv_searcher
years = [str(y) for y in [2013, 2014, 2015, 2016]]

fn_ins = [os.path.join(wd, 'cvc_{year}.h5'.format(year=year)) for year in years]
model_file = os.path.join(od, '{name}_{years}_weights.h5').format(name=model_func.__name__,
                                                                  years="_".join(years))

dp = DataProcessor(exclude_n=10, exclude_capture=True, verbosity=VERBOSITY.VERBOSE)
dp.load_many_h5(fn_ins)

print("Total Dataset size:")
print("\tWin Moves: %s" % dp.win_boards)
print("\tLoss Moves: %s" % dp.loss_boards)

traingen, valgen = dp.create_gen(batch_size=BATCH_SIZE, test_split=0.15, random_state=2346, flat=False)

# for i in range(100):
    # print("Train Gen")
    # next(traingen)
    # print("Val Gen")
    # next(valgen)
    # input("Continue?")

optimizer = SGD(lr=0.0085, momentum=0.9, nesterov=True, decay=0.01)
optimizer = Adam(lr=0.0001)
# This will save the best scoring model weights to the parent directory
best_model = ModelCheckpoint(model_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True,
                             save_weights_only=True)
plotter = Plotter()
model = model_func(optimizer)
model.load_weights(model_file)
print(model.summary())

print("Fitting model")
fit = model.fit_generator(traingen, steps_per_epoch=EPOCH_STEPS,
                          epochs=200, verbose=1, callbacks=[best_model, plotter],
                          validation_data=valgen, validation_steps=VAL_STEPS,
                          initial_epoch=13)
