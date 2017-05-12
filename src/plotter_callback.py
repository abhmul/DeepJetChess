import keras
import matplotlib.pyplot as plt
import numpy as np

class Plotter(keras.callbacks.Callback):

    def __init__(self):
        super().__init__()
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.x = []
        self.y_train = []
        self.y_val = []
        # self.ax.plot(self.x, self.y_train, 'b-', self.x, self.y_val, 'g-')

    def on_train_end(self, logs={}):
        plt.ioff()
        plt.show()
        return

    def on_epoch_end(self, epoch, logs={}):
        # self.line1.set_ydata(logs.get('loss'))
        # self.line2.set_ydata(logs.get('val_loss'))
        self.x.append(len(self.x))
        self.y_train.append(logs.get('loss'))
        self.y_val.append(logs.get('val_loss'))
        self.ax.clear()
        self.ax.plot(self.x, self.y_train, 'b-', self.x, self.y_val, 'g-')
        self.fig.canvas.draw()
        # plt.pause(0.5)
        return
