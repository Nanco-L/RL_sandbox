import tensorflow as tf
import numpy as np
import agent.model as am
import board.board as bd
import agent.agent as ag

fc = am.FCN(9,10)
dqn = am.DQNWrapper(fc)

myboard = bd.Tictactoe()
H1 = ag.Human()
H2 = ag.Human()
B1 = ag.DQNBot(dqn)
B2 = ag.DQNBot(dqn)

for i in range(1000):
    myboard.run(B1, B2, verbose=False)
    #print(len(B1.replay))

B1.generate_dataset()

dqn.run(2000, B1.train_ds)

myboard.run(B1, B2, verbose=True)

"""
state  = np.random.random([10,9])
action = (np.random.random([10,1])*10).astype(np.int)
value  = np.random.random([10,1])

train_ds = tf.data.Dataset.from_tensor_slices((state, action, value)).shuffle(100).batch(2)

dqn.run(2000, train_ds)
"""

