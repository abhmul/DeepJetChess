# DeepJetChess

This is a Chess Bot that I made with deep learning that can play at a 1650 elo (top 4%) level. The architecture is my Dynamic Policy Net architecture, which computes a scalar score for each next legal board position and takes a softmax over those scores to get a probability distribution over all the possible moves it can make.

# Installation

This bot depends on the following packages
```
numpy
matplotlib
python-chess
h5py
pytorch
pyjet
```

The first 4 packages can be installed through `pip` like so:
```
pip3 install PACKAGE_NAME
```
Installation for pytorch is included in their [website](http://pytorch.org/). PyJet is my own custom front-end I wrote for PyTorch so I could use a Keras-like API for PyTorch without sacrificing its flexibility. Installation instructions are included in the [PyJet Repo](https://github.com/abhmul/PyJet).

To get GPU speedup, you'll want to install CUDA and CudNN. If you're working off of an AWS EC2 machine or Google Cloud Compute Machine, you can use my [3-step installation](https://github.com/abhmul/InstallCUDA-Kerasv2).

# Compiling the Dataset

You'll want to download pgn files of games to train the network on. I used games played by humans with >2000 elo from the [FICS Games Database](http://www.ficsgames.org/download.html). Once you've downloaded all the games you want, run the following command:
```
python3 read_games.py PATH_TO_H5_OUT PATH_TO_EACH_PGN_FILE...
```

# Training

Once you have the dataset ready, you'll want to create a folder `DeepJetChess/models` to save models in. You'll also need to modify the `FN_IN` global in the `train.py` file to whatever you saved your compiled H5 dataset to. Once you've made the modifications you can run
```
python3 train.py
```

# Reinforcement Learning

Once you've initialized your model with the supervised learning on human games, you'll need to modify line 187 in `reinforce.py` to load whatever the name of your saved supervised learning model is from the training step. Then you can just run
```
python3 reinforce.py
```

Each successive checkpoint will be saved as `PrevNeti.state` where `i` is the checkpoint number.

# Elo Testing

To test the model's elo, you can change the `MODEL_NAME` global (line 66) in `elo_test.py` to whatever model you want to test. Then run
```
python3 elo_test.py
```
and the script will print the moves corresponding to [this online elo test](http://www.chessmaniac.com/ELORating/ELO_Chess_Rating.shtml) where you'll have to manually enter the moves.
