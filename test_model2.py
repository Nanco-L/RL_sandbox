import numpy as np
import board.board as bd
import agent.agent as ag

myboard = bd.Tictactoe()
H1 = ag.Human()
H2 = ag.Human()
B1 = ag.QNBot()
B2 = ag.QNBot()

for i in range(50000):

    myboard.run(B1, B2, verbose=False)

    if B1.exploration_rate > 0.3:
        B1.exploration_rate -= 0.001
        B2.exploration_rate -= 0.001

    if i%1000 == 0:
        print(B1.q_dict.get('211220220', 0.0))

myboard.run(B1, B2, verbose=True)
#print(B1.q_dict)
