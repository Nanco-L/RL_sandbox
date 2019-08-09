import numpy as np

class Human():
    def __init__(self):
        self.result_log = list()
        
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
        
    def save_result(self, result):
        self.result_log.append(result)

class Bot():
    def __init__(self):
        self.result_log = list()
        self.value_function = dict()
        self.exploration_rate = 0.9
        self.decaying_rate = 0.9
        self.learning_rate = 0.5
        self.trajectory = list()
        self.trainable = True
        
    def get_move(self, state, avail_action):
        # choose the state with maximum value
        state = ''.join(str(item) for item in state)
        #print(self.value_function.keys(), state)
        if state in self.value_function.keys():
            if self.exploration_rate < np.random.random():
                #action = self.value_function[state][sorted(self.value_function[state], key=self.value_function[state].get)[0]]
                action = sorted(self.value_function[state], key=self.value_function[state].get)[-1]
            else:
                #print("explore")
                action = np.random.choice(np.arange(len(avail_action))[avail_action], 1)[0]
        else:
            self.add_new_state_in_value_function(state)
            action = np.random.choice(np.arange(len(avail_action))[avail_action], 1)[0]
            
        self.trajectory.append((state, action))
        return action
    
    def serious_mode(self):
        self.exploration_rate = 0.0
        
    def train_mode(self, exploration_rate=0.9):
        self.exploration_rate = exploration_rate
        
    def fix_model(self):
        self.trainable = False
        
    def free_model(self):
        self.trainable = True
        
    def save_result(self, result):
        self.result_log.append(result)
        if self.trainable:
            self.update_value_function(result)
        
    def update_value_function(self, result):
        new_state, new_action = self.trajectory.pop()
        #print(self.value_function[new_state])
        self.value_function[new_state][new_action] = result
        
        result *= self.decaying_rate
        
        while self.trajectory:
            state, action = self.trajectory.pop()
            #print(self.value_function[state])
            self.value_function[state][action] += \
                self.learning_rate * (result + self.decaying_rate*self.value_function[new_state][new_action] - self.value_function[state][action])
        
            result *= self.decaying_rate
        
            new_state = state
            new_action = action
            
    def add_new_state_in_value_function(self, state):
        avail_keys = np.array([item for item in state]) == '2'
        
        self.value_function[state] = dict()
        
        for item in np.arange(len(state))[avail_keys]:
            self.value_function[state][item] = 0.