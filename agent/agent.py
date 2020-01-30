import numpy as np
import tensorflow as tf

class Agent():
    def __init__(self):
        self.result_log = list()
        self.player_number = 0

    def get_move(self):
        """
        Override
        """

    def set_player_number(self, number):
        self.player_number = number

    def save_result(self, result, playlog):
        self.result_log.append([playlog, result])

class Human(Agent):
        
    def get_move(self, state, avail_action):
        while True:
            action = input("Next_move:")
            try:
                action = int(action)
            except:
                print("Please type number")
                continue
            
            if action > len(avail_action) or action < 1:
                print("Please type number between 1~{}".format(len(avail_action)))
            elif avail_action[action-1]:
                break
            else:
                print("That tile is already filled. please type unfilled tile (" + 
                      ", ".join((np.arange(len(avail_action))[avail_action]+1).astype(np.str)) + ")")
            
        return action-1

class DQNBot(Agent):
    def __init__(self, model):
        super(DQNBot, self).__init__()
        self.model = model
        self.replay = list()
        self.decay_rate = 0.9
        self.exploration_rate = 1.

    def get_move(self, state, avail_action):
        if np.random.random() < self.exploration_rate:
            action_list = np.arange(avail_action.shape[0])[avail_action]
            np.random.shuffle(action_list)
    
            return action_list[0]
        else:
            values = self.model.model(state)
        
            return np.argmax(values[avail_action])

    def save_result(self, result, playlog):

        s_ = None
        while playlog:
            i = len(playlog)
            s, a, _ = playlog.pop()
            
            s = np.array(s)
            if self.player_number == 0:
                s = (s+2)%3 - 1
            elif self.player_number == 1:
                s = (s*2)%3 - 1
            s = s.astype(np.float)
                
            if (i+1)%2 == self.player_number:
                self.replay.append([s, a, result, s_])
                s_ = s

            result *= self.decay_rate

    def generate_dataset(self):
        
        state  = list()
        action = list()
        value  = list()

        for item in self.replay:
            state.append(item[0])
            action.append(item[1])
            if item[3] is None:
                value.append(item[2])
            else:
                one_hot = np.zeros([9])
                one_hot[item[1]] = 1
                value.append(item[2] + np.sum(self.decay_rate*self.model.model(np.array([item[3]]))*one_hot))

        self.train_ds = tf.data.Dataset.from_tensor_slices((state, action, value)).shuffle(2000).batch(16)

    def train(self):
        self.model.run()

        
