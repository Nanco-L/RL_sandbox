import tensorflow as tf
import numpy as np
import agent.model as am
import board.board as bd
import agent.agent as ag

fc = am.FCN(9,30)
dqn = am.DQNWrapper(fc)

myboard = bd.Tictactoe()
H1 = ag.Human()
H2 = ag.Human()
B1 = ag.DQNBot(dqn)
B2 = ag.DQNBot(dqn)

B1.exploration_rate = 0.9
B2.exploration_rate = 0.9

global_epoch = 0
for ii in range(1000):
    #print(f'=== {ii+1:4d} th generation ===')

    for i in range(5):
        myboard.run(B1, B2, verbose=False)
        #print(len(B1.replay))

    tset1 = B1.generate_dataset()
    tset2 = B2.generate_dataset()

    #global_epoch = dqn.run(50, tset1.concatenate(tset2).shuffle(10000).batch(16), global_epoch)
    global_epoch = dqn.run(5, tset1.shuffle(10000).batch(16), global_epoch)

    if (ii+1)%50 == 0:
        if B1.exploration_rate > 0.1:
            B1.exploration_rate -= 0.2
            print(f'exploration_rate: {B1.exploration_rate}')
        if B2.exploration_rate > 0.1: 
            B2.exploration_rate -= 0.2
        print(f'{dqn.model(np.array([[1.,-1.,-1.,1.,1.,-1.,-1.,1.,0.]]))}, {np.sum(dqn.model(np.array([[1.,-1.,-1.,1.,1.,-1.,-1.,1.,0.]])).numpy())}')
        print(f'{dqn.model(np.array([[0.,-1.,-1.,0.,0.,1.,0.,0.,1.]]))}, {np.sum(dqn.model(np.array([[0.,-1.,-1.,0.,0.,1.,0.,0.,1.]])).numpy())}')
        dqn.save()

    #B1.clear_replay()
    #B2.clear_replay()

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

