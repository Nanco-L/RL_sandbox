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

#B1.exploration_rate -= 0.15
#B2.exploration_rate -= 0.15

for ii in range(20):
    print(f'=== {ii+1:4d} th generation ===')

    for i in range(1000):
        myboard.run(B1, B2, verbose=False)
        #print(len(B1.replay))

    B1.generate_dataset(others=[B2.replay])

    dqn.run(100, B1.train_ds)

    if B1.exploration_rate > 0.25:
        B1.exploration_rate -= 0.15
    if B2.exploration_rate > 0.25:
        B2.exploration_rate -= 0.15

    B1.clear_replay()
    B2.clear_replay()

B1.exploration_rate = 0.
B2.exploration_rate = 0.

myboard.run(B1, B2, verbose=True)

dqn.save()

"""
state  = np.random.random([10,9])
action = (np.random.random([10,1])*10).astype(np.int)
value  = np.random.random([10,1])

train_ds = tf.data.Dataset.from_tensor_slices((state, action, value)).shuffle(100).batch(2)

dqn.run(2000, train_ds)
"""

