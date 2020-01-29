import board.board as bd
import agent.agent as ag

myboard = bd.Tictactoe()
H1 = ag.Human()
H2 = ag.Human()
B1 = ag.DQNBot(None)
B2 = ag.DQNBot(None)

for i in range(3):
    myboard.run(B1, B2, verbose=False)
    print(len(B1.replay))

#print(B1.replay)
#print(B2.replay)
