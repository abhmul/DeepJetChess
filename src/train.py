from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import dataprocessor as dp
import models

model_func = models.relu3layer
graphing = True

fn_in = '../chess_games_2016.h5'
X_before, X_after = dp.load_h5(fn_in, shuffle=True)
X_before_train, X_before_test, X_after_train, X_after_test = train_test_split(X_before, X_after, train_size=.85)

traingen = dp.chessgen(X_before_train, X_after_train, selection_labels=True,
                       movement_labels=False, split=12, batch_size=32,
                       shuffle=True)
valgen = dp.chessgen(X_before_test, X_after_test, selection_labels=True,
                     movement_labels=False, split=12, batch_size=32,
                     shuffle=True)

optimizer = SGD(lr=0.005, momentum=0.9, decay=0.98, nesterov=True)

# This will save the best scoring model weights to the parent directory
best_model_file = '../models/relu3layer_weights.h5'
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=1, save_best_only=True,
                             save_weights_only=True)

model = model_func(optimizer=optimizer)
fit = model.fit_generator(traingen, samples_per_epoch=X_before_train.shape[0],
                          nb_epoch=100, verbose=1, callbacks=[best_model],
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
