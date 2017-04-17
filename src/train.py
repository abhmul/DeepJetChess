from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD, RMSprop
from dataprocessor3 import DataProcessor
import models

import os

FLOYD = False
if FLOYD:
    wd = "/input"
    od = "/output"
else:
    wd = "../"
    od = "../models"

model_func = models.conv4layer

movement_labels = False
movements = 'movements_' if movement_labels else ''

cvc = True
cls = 'cvc_' if cvc else ''

years = [2015, 2016]
exclude_n = 5
prev_boards = 4
best_model_file = os.path.join(od, '{name}_{cls}{years}_{movements}exclude{exclude_n}_weights.h5')

best_model_file = best_model_file.format(name=model_func.__name__,
                                         cls=cls,
                                         years=years,
                                         movements=movements,
                                         exclude_n=exclude_n)
graphing = not FLOYD
num_channels = 12
batch_size=32

fn_ins = [os.path.join(wd, 'chess_games_{cls}{year}.h5'.format(cls=cls, year=year)) for year in years]
dp = DataProcessor(movement_labels=movement_labels, exclude_n=exclude_n, prev_boards=prev_boards, validation_split=0.15)
for fn_in in fn_ins:
    dp.load_h5(fn_in)

print("Total Dataset size: {} moves and {} games".format(dp.num_moves, dp.num_games))

traingen = dp.chessgen(batch_size=batch_size, shuffle=True, validation=False)
valgen = dp.chessgen(batch_size=batch_size, shuffle=True, validation=True)

optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
rms = RMSprop(lr=.0015, decay=.999)
# optimizer = 'adam'

# This will save the best scoring model weights to the parent directory
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True,
                             save_weights_only=True)

model = model_func(num_channels=num_channels, prev_boards=prev_boards, optimizer=optimizer, movement_labels=movement_labels)
print(model.summary())
# model.load_weights(best_model_file)

def scheduler(epoch):
    if epoch < 30:
        print("Learning Rate for Epoch %s: %s" %(epoch, 0.001))
        return float(0.001)
    elif epoch < 50:
        print("Learning Rate for Epoch %s: %s" %(epoch, 0.0001))
        return float(0.0001)
    elif epoch < 60:
        print("Learning Rate for Epoch %s: %s" %(epoch, 0.00001))
        return float(0.00001)
    else:
        print("Learning Rate for Epoch %s: %s" %(epoch, 0.000001))
        return float(0.000001)


change_lr = LearningRateScheduler(scheduler)
steps_per_epoch = int((dp.train_size + batch_size - 1) / batch_size)
val_steps = int((dp.val_size + batch_size - 1) / batch_size)

print("Fitting model")
fit = model.fit_generator(traingen, steps_per_epoch=steps_per_epoch,
                          epochs=50, verbose=1, callbacks=[best_model, change_lr],
                          validation_data=valgen, validation_steps=val_steps)

if graphing:
    import matplotlib.pyplot as plt

    plt.plot(fit.history['val_loss'])
    plt.title('Validation Loss')
    plt.show()

    plt.plot(fit.history['val_acc'])
    plt.title('Validation Accuracy')
    plt.show()

    plt.plot(fit.history['val_top_k_categorical_accuracy'])
    plt.title('Validation Top-3 Accuracy')
    plt.show()
else:
    print('Validation Loss')
    print(fit.history['val_loss'])

    print('Validation Accuracy')
    print(fit.history['val_acc'])
