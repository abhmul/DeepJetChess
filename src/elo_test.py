import os
import chess
import chess.pgn
from torch_models import AlphaChess
from players import Net

def setup_game(fen):
    b = chess.Board(fen=fen)
    g = chess.pgn.Game()
    g.setup(b)
    return g

# Board 1
fen1 = "r1b3k1/6p1/P1n1pr1p/q1p5/1b1P4/2N2N2/PP1QBPPP/R3K2R b - - 0 1"
GAME1 = setup_game(fen1)

# Board2
fen2 = "2nq1nk1/5p1p/4p1pQ/pb1pP1NP/1p1P2P1/1P4N1/P4PB1/6K1 w - - 0 1"
GAME2 = setup_game(fen2)

# Board3
fen3 = "8/3r2p1/pp1Bp1p1/1kP5/1n2K3/6R1/1P3P2/8 w - - 0 1"
GAME3 = setup_game(fen3)

# Board4
fen4 = "8/4kb1p/2p3pP/1pP1P1P1/1P3K2/1B6/8/8 w - - 0 1"
GAME4 = setup_game(fen4)

# Board5
fen5 = "b1R2nk1/5ppp/1p3n2/5N2/1b2p3/1P2BP2/q3BQPP/6K1 w - - 0 1"
GAME5 = setup_game(fen5)

# Board6
fen6 = "3rr1k1/pp3pbp/2bp1np1/q3p1B1/2B1P3/2N4P/PPPQ1PP1/3RR1K1 w - - 0 1"
GAME6 = setup_game(fen6)

# Board7
fen7 = "r1b1qrk1/1ppn1pb1/p2p1npp/3Pp3/2P1P2B/2N5/PP1NBPPP/R2Q1RK1 b - - 0 1"
GAME7 = setup_game(fen7)

# Board8
fen8 = "2R1r3/5k2/pBP1n2p/6p1/8/5P1P/2P3P1/7K w - - 0 1"
GAME8 = setup_game(fen8)

# Board9
fen9 = "2r2rk1/1p1R1pp1/p3p2p/8/4B3/3QB1P1/q1P3KP/8 w - - 0 1"
GAME9 = setup_game(fen9)

# Board10
fen10 = "r1bq1rk1/p4ppp/1pnp1n2/2p5/2PPpP2/1NP1P3/P3B1PP/R1BQ1RK1 b - - 0 1"
GAME10 = setup_game(fen10)

GAMES = (GAME1, GAME2, GAME3, GAME4, GAME5, GAME6, GAME7, GAME8, GAME9, GAME10)
assert(len(GAMES) == 10)

def elo_test(player):
    moves = [player.move(gn).move for gn in GAMES]
    for i, move in enumerate(moves):
        print("Diagram %s: %s" %(i+1, move))

if __name__ == "__main__":
    SOURCE = os.getcwd()
    WORKING = os.path.join(SOURCE, '..')
    OUTPUT = os.path.join(WORKING, 'models')
    # for EPOCH_NUM in range(20):
    MODEL_NAME = "PrevNet3.state"
    # MODEL_NAME = "alpha_chess2_epoch{}_weights.state".format(EPOCH_NUM)
    # Setup the model
    alpha_chess = AlphaChess()
    player = Net(alpha_chess, name="AlphaChess")
    print("Running Elo Test for %s" % player.name)
    print("Loading the model: ", MODEL_NAME)
    player.load_state(os.path.join(OUTPUT, MODEL_NAME))
    print("Model Loaded!")
    # for i in range(10):
        # print("Trial %s" % i)
    elo_test(player)
