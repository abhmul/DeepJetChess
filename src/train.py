from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD, RMSprop
import dataprocessor as dp
import models

model_func = models.conv4layer
movement_labels = True
cvc = True
year = 2015
exclude_n = 5
best_model_file = '../models/conv4layer_{}{}_{}exclude{}_weights.h5'
if movement_labels:
    to_load = best_model_file.format('cvc_' if cvc else '_',
                                     2016,
                                     '', exclude_n)
best_model_file = best_model_file.format('cvc_' if cvc else '_',
                                         year,
                                         'movements_' if movement_labels
                                         else '', exclude_n)
graphing = True
num_channels = 12

fn_in = '../chess_games_{}{}.h5'.format('cvc_' if cvc else '_', year)
X_before, X_after = dp.load_h5(fn_in, shuffle=False, exclude_n=exclude_n)
X_before_train, X_before_test, X_after_train, X_after_test = train_test_split(X_before, X_after, train_size=.85)

traingen = dp.chessgen(X_before_train, X_after_train, selection_labels=True,
                       movement_labels=movement_labels, split=num_channels, prev_boards=0, batch_size=32,
                       shuffle=True)
valgen = dp.chessgen(X_before_test, X_after_test, selection_labels=True,
                     movement_labels=movement_labels, split=num_channels, batch_size=32,
                     shuffle=True)
# next(traingen)
# raise StopIteration

optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
rms = RMSprop(lr=.0015, decay=.999)

# optimizer = None

# This will save the best scoring model weights to the parent directory
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True,
                             save_weights_only=True)

model = model_func(num_channels=num_channels, optimizer=optimizer, movement_labels=movement_labels, to_load=to_load)
print(model.summary())
# model.load_weights(best_model_file)

def scheduler(epoch):
    if epoch < 15:
        print("Learning Rate for Epoch %s: %s" %(epoch, 0.001))
        return float(0.001)
    elif epoch < 30:
        print("Learning Rate for Epoch %s: %s" %(epoch, 0.0001))
        return float(0.0001)
    elif epoch < 60:
        print("Learning Rate for Epoch %s: %s" %(epoch, 0.00001))
        return float(0.00001)
    else:
        print("Learning Rate for Epoch %s: %s" %(epoch, 0.000001))
        return float(0.000001)


change_lr = LearningRateScheduler(scheduler)

fit = model.fit_generator(traingen, samples_per_epoch=X_before_train.shape[0],
                          nb_epoch=50, verbose=1, callbacks=[best_model, change_lr],
                          validation_data=valgen, nb_val_samples=X_before_test.shape[0])

if graphing:
    import matplotlib.pyplot as plt

    plt.plot(fit.history['val_loss'])
    plt.title('Validation Loss')
    plt.show()

    plt.plot(fit.history['val_acc'])
    plt.title('Validation Accuracy')
    plt.show()
else:
    print('Validation Loss')
    print(fit.history['val_loss'])

    print('Validation Accuracy')
    print(fit.history['val_acc'])
