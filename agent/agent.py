import numpy as np
import tensorflow as tf

class Agent():
    def __init__(self):
        self.replay = list()
        self.player_number = 0

    def get_move(self):
        """
        Override
        """

    def set_player_number(self, number):
        self.player_number = number

    def save_result(self, result, playlog):
        self.replay.append([playlog, result])

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

# TODO: Need to make replay buffer

class DQNBot(Agent):
    def __init__(self, model):
        super(DQNBot, self).__init__()
        self.model = model
        #self.replay = list()
        self.decay_rate = 0.9
        self.exploration_rate = 1.
        self.buffer_size = 1000

    def get_move(self, state, avail_action):
        if np.random.random() < self.exploration_rate:
            action_list = np.arange(avail_action.shape[0])[avail_action]
            np.random.shuffle(action_list)
    
            return action_list[0]
        else:
            state_fix = (np.copy(state)+2)%3 - 1

            #print(state)
            #print(self.model.model(np.array([state_fix])).numpy()[0])
            values = self.model.model(np.array([state_fix])).numpy()[0][avail_action]
                        
            if self.player_number == 0:
                return np.arange(len(state))[avail_action][np.argmax(values)]
            else:
                return np.arange(len(state))[avail_action][np.argmin(values)]

    def save_result(self, result, playlog):

        #s_ = None
        max_len = len(playlog)
        while playlog:
            i = len(playlog)
            s, a, s_, avail_action = playlog.pop()

            s = (s+2)%3 - 1
            s_ = (s_+2)%3 - 1

            if max_len == i:
                q = result
            else:
                if i%2 == 0:
                    q = np.min(self.model.model(np.array([s_])).numpy()[0][avail_action])
                else:
                    q = np.max(self.model.model(np.array([s_])).numpy()[0][avail_action])

            self.replay.append([s, a, result, s_, q])
            if len(self.replay) > self.buffer_size:
                self.replay.pop(0)
        
        """
        s, a, s_, avail_action = playlog.pop()

        s = (s+2)%3 - 1
        s_ = (s_+2)%3 - 1

        q = result
        self.replay.append([s, a, result, s_, q])
        if len(self.replay) > self.buffer_size:
            self.replay.pop(0)

        while playlog:
            #i = len(playlog)
            s, a, s_, avail_action = playlog.pop()

            s = (s+2)%3 - 1
            s_ = (s_+2)%3 - 1

            #if (i+1)%2 == self.player_number:
            # TODO: value calculation
            
            #if np.sum(avail_action) == 1:
            #    q = result
            #else:
                #v = result + self.decay_rate*self.model.model(np.array([s_]).numpy()[0,a]
                #v = result + self.decay_rate*np.max(self.model.model(np.array([s_]).numpy()))#[0,a]
                #v = result - self.decay_rate*np.max(self.model.model(np.array([s_])).numpy())
            
            q = self.decay_rate * np.max(self.model.model(np.array([s_])).numpy()[0][avail_action]) # avail_action

            self.replay.append([s, a, result, s_, q])
            if len(self.replay) > self.buffer_size:
                self.replay.pop(0)
            #s_ = s
        """
            

    def generate_dataset(self):
        
        state  = list()
        action = list()
        value  = list()

        for item in self.replay:
            state.append(item[0])
            action.append(item[1])
            value.append(item[4])
            
            """
            if item[3] is None:
                value.append(item[2])
            else:
                one_hot = np.zeros([9])
                one_hot[item[1]] = 1
                value.append(item[2] + np.sum(self.decay_rate*self.model.model(np.array([item[3]]))*one_hot))
            """

        return tf.data.Dataset.from_tensor_slices((state, action, np.array(value, dtype=np.float))) #.shuffle(10000).batch(16)

    def clear_replay(self):
        self.replay = list()

    def train(self):
        self.model.run()

        
