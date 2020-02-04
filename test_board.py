import tensorflow as tf
import numpy as np
import agent.model as am
import board.board as bd
import agent.agent as ag

fc = am.FCN(9,10)
dqn = am.DQNWrapper(fc)
#dqn.load()

myboard = bd.Tictactoe()
H1 = ag.Human()
H2 = ag.Human()
B1 = ag.DQNBot(dqn)
B2 = ag.DQNBot(dqn)

B1.exploration_rate = 0.
B2.exploration_rate = 0.

for i in range(1):
    myboard.run(B1, B2, verbose=True)
    print(B1.replay)

#print(B1.replay)
#print(B2.replay)
