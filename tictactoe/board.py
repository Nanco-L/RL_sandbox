import numpy as np

class Board():
    def __init__(self, size=3):
        self.size_of_board = size
        self.board = None
        self.avail_actions = None
        self.reset_board()
        self.get_avail_actions()
        
        self.turn = 0
        self.stale = False
        
        self.players = [None, None]
        self.mark = {2:' ', 0:'X', 1:'O'}
        
        self.current_player = 0
        self.pickone_board = None
        
        hori_line = "{}"+" | {}"*(self.size_of_board-1)+"\n"
        split_line = "---"+"----"*(self.size_of_board-1)+"\n"
        self.board_tile = hori_line
        
        for _ in range(self.size_of_board-1):
            self.board_tile += split_line + hori_line
        
    def reset_board(self):
        self.board = np.array([2]*self.size_of_board*self.size_of_board, dtype=np.int8)
        self.get_avail_actions()
    
    def get_avail_actions(self):
        self.avail_actions = self.board == 2
        
    def set_players(self, player1, player2):
        self.players[0] = player1
        self.players[1] = player2
    
    def update_board(self):
        self.board[self.players[self.current_player].get_move(self.board, self.avail_actions)] = self.current_player
        self.get_avail_actions()
        #print(self.avail_actions)
                
    #def get_user_move(self):
    #    while True:
            #user_action = int(input("Next move:"))
    #        user_action = self.players[self.current_player].get_move(self.avail_actions)
            
    #        if user_action > self.size_of_board*self.size_of_board-1 or user_action < -1:
    #            print("The action is not allowed")
    #        elif self.avail_actions[user_action]:
    #            break
    #        else:
    #            print("The action is not allowed")
                
    #    return user_action
    
    def print_board(self):
    #    print("""
    # {} | {} | {}
    #-----------
    # {} | {} | {}
    #-----------
    # {} | {} | {}
    # """.format(*([self.mark[i] for i in self.board])))
         
        print(self.board_tile.format(*([self.mark[i] for i in self.board])))
        
    def check_winner(self):
        # use numpy
        self.pickup_board = self.board == self.current_player
        if (np.sum(self.pickup_board)) < self.size_of_board:
            return False
        else:
            for item in self.board.reshape([self.size_of_board, self.size_of_board]):
                if np.sum(item == self.current_player) == self.size_of_board:
                    return True
                
            for item in self.board.reshape([self.size_of_board, self.size_of_board]).T:
                if np.sum(item == self.current_player) == self.size_of_board:
                    return True
                
            if np.sum(self.board.reshape(
                [self.size_of_board, self.size_of_board]
            ).diagonal() == self.current_player) == self.size_of_board:
                return True
            
            if np.sum(np.fliplr(self.board.reshape(
                [self.size_of_board, self.size_of_board])
            ).diagonal() == self.current_player) == self.size_of_board:
                return True
            
        return False
        
        
    def run(self, player1, player2, verbose=True):
        self.reset_board()
        self.current_player = 0
        self.set_players(player1, player2)
        self.turn = 1
        
        while True:
            #clear_output()
            self.update_board()
            if verbose:
                print('===== Turn {} (Player {}) ====='.format(self.turn, self.current_player+1))
                self.print_board()
            if self.check_winner():
                if verbose:
                    print("Player {} win!".format(self.current_player+1))
                self.players[self.current_player].save_result(1)
                self.players[(self.current_player+1)%2].save_result(-1)
                break
            elif np.sum(self.avail_actions) == 0:
                if verbose:
                    print("Draw!")
                self.players[self.current_player].save_result(-0.5)
                self.players[(self.current_player+1)%2].save_result(-0.5)
                break
            self.current_player = (self.current_player + 1)%2
            self.turn += 1