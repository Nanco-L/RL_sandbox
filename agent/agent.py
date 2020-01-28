import numpy as np
import tensorflow as tf

class Agent():
    def __init__(self):
        self.result_log = list()

    def get_move(self):
        """
        Override
        """

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
        super().__init__()
        self.model = model
        self.replay = list()
        self.player_number = 0
        self.decay_rate = 0.9

    def set_player_number(self, number):
        self.player_number = number

    def get_move(self, state, avail_action):
        values = self.model.model(state)
        action = np.max(values[avail_action])
        
        return np.max(values[avail_action])

    def save_result(self, result, playlog):

        s_ = None
        while playlog:
            i = len(playlog)

            if i%2 == self.player_number:
                s, a, _ = playlog.pop()
                self.replay.append([s, a, result, s_])
                s_ = s

            result *= self.decay_rate

        
