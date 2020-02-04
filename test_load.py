import tensorflow as tf
import numpy as np
import agent.model as am
import board.board as bd
import agent.agent as ag

fc = am.FCN(9,10)
dqn = am.DQNWrapper(fc)
dqn.load()

myboard = bd.Tictactoe()
H1 = ag.Human()
H2 = ag.Human()
B1 = ag.DQNBot(dqn)
B2 = ag.DQNBot(dqn)

B1.exploration_rate = 0.
B2.exploration_rate = 0.

myboard.run(B1, H2, verbose=True)
myboard.run(H1, B2, verbose=True)
