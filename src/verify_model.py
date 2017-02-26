import dataprocessor as dp
import models
import tensorflow as tf
import numpy as np

with tf.device('/cpu:0'):
    model_func = models.conv4layer

    fn_in = '../chess_games_2016.h5'
    X_before, X_after = dp.load_h5(fn_in, shuffle=True)

    datagen = dp.chessgen(X_before, X_after, selection_labels=True,
                          movement_labels=False, split=12, batch_size=1,
                          shuffle=True, debug=True)

    model = model_func()
    model.load_weights('../models/conv4layer_cvc_2016_exclude5_weights.h5')

    for x_batch, y_batch in datagen:
        pred = model.predict_on_batch(x_batch)[0]
        print('Predictions for piece selection:\n')
        print(np.around(pred, decimals=3))
        input('Continue?')
