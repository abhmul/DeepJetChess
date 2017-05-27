import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD, RMSprop, Adam
import tensorflow as tf
from dataprocessor5 import DataProcessor, VERBOSITY
import models


from plotter_callback import Plotter

BATCH_SIZE = 32
EPOCH_STEPS = int((1000000 + BATCH_SIZE - 1) / BATCH_SIZE)
# EPOCH_STEPS = 1
VAL_STEPS = int(0.15 * EPOCH_STEPS) + 1
INCLUDE_RIGHTS = False
#

sd = os.getcwd()
wd = os.path.join(sd, '..')
od = os.path.join(wd, 'models')

model_func = models.conv_comparator
distilled_func = models.distilled_conv_comparator
years = [str(y) for y in [2013, 2014, 2015, 2016]]

fn_ins = [os.path.join(wd, 'cvc_{year}.h5'.format(year=year)) for year in years]
model_file = os.path.join(od, '{name}_{years}_weights.h5').format(name=model_func.__name__,
                                                                  years="_".join(years))
distilled_file = os.path.join(od, '{name}_{years}_weights.h5').format(name=distilled_func.__name__,
                                                                  years="_".join(years))

dp = DataProcessor(exclude_n=10, exclude_capture=True, include_rights=INCLUDE_RIGHTS, verbosity=VERBOSITY.VERBOSE)
dp.load_many_h5(fn_ins)

print("Total Dataset size:")
print("\tWin Moves: %s" % dp.win_boards)
print("\tLoss Moves: %s" % dp.loss_boards)

dp.create_gen(batch_size=BATCH_SIZE, test_split=0.15, random_state=2346, flat=False)

# optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
# optimizer = Adam(lr=0.00001)
optimizer=None

# This will save the best scoring model weights to the parent directory
best_model = ModelCheckpoint(distilled_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True,
                             save_weights_only=True)
plotter = Plotter()
model = model_func(optimizer, include_rights=INCLUDE_RIGHTS, mode='distill')
model.load_weights(model_file)
# model.save_weights(model_file) # Uncomment this to rename layers
# Allow us to generate data asynchrounoulsy
graph = tf.get_default_graph()
global graph

distilled = distilled_func(optimizer, include_rights=INCLUDE_RIGHTS, mode='distill')

print(model.summary())
print(distilled.summary())

print("Fitting model")
fit = distilled.fit_generator(dp.distillgen(model, 'train', graph=graph), steps_per_epoch=EPOCH_STEPS,
                          epochs=200, verbose=1, callbacks=[best_model, plotter],
                          validation_data=dp.distillgen(model, 'val', graph=graph), validation_steps=VAL_STEPS)
                        #   initial_epoch=28)
