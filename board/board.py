import numpy as np

class Board():
    """
    Basic class for gameborad.
    This board has square shape and has 3 state ('X': player 1, 'O': player 2, ' ': empty)
    """
    def __init__(self, size=3):
        """
        size (int, default: 3): Size of board. The shape of board is [size, size]
        """
        self.size_of_board = size
        hori_line = " {}"+" | {}"*(self.size_of_board-1)+"\n"
        split_line = "---"+"----"*(self.size_of_board-1)+"\n"
        self.board_tile = hori_line
        
        for _ in range(self.size_of_board-1):
            self.board_tile += split_line + hori_line

        # FIXME: check dimension
        self._reset_state()

        self.turn = 1
        self.players = [None, None]
        self.current_player = 0
        self.current_move = None

        self.mark = {0: 'X', 1: 'O', 2: ' '}

        self.playlog = list()

    def _reset_state(self):
        self.state = np.full([self.size_of_board*self.size_of_board], 2, dtype=np.int8)
        self._get_avail_actions()

    def _get_avail_actions(self):
        self.avail_actions = self.state == 2

    def _render(self):
        print(self.board_tile.format(*([self.mark[i] for i in self.state])))

    def _check_winner(self):
        """
        Override this function for generating specific winning condition
        """
        return 0

    def _update_state(self):
        self.current_move = self.players[self.current_player].get_move(self.state, self.avail_actions)
        self.state[self.current_move] = self.current_player
        self._get_avail_actions()
        return self.current_move

    def run(self, player1, player2, verbose=False):
        self._reset_state()

        self.turn = 1
        self.players = [player1, player2]
        self.players[0].set_player_number(0)
        self.players[1].set_player_number(1)
        self.current_player = 0
        self.current_move = None
        self.playlog = list()

        while True:
            state = np.copy(self.state)
            action = self._update_state()
            self.playlog.append([state, action, np.copy(self.state)])
            if verbose:
                print('===== Turn {} (Player {}) ====='.format(self.turn, self.current_player+1))
                self._render()
            if self._check_winner():
                if verbose:
                    print("Player {} win!".format(self.current_player+1))
                self.players[self.current_player].save_result(1, self.playlog[:])
                self.players[(self.current_player+1)%2].save_result(-1, self.playlog[:])
                break
            elif np.sum(self.avail_actions) == 0:
                if verbose:
                    print("Draw!")
                self.players[self.current_player].save_result(-0.5, self.playlog[:])
                self.players[(self.current_player+1)%2].save_result(-0.5, self.playlog[:])
                break
            self.current_player = (self.current_player + 1)%2
            self.turn += 1

class Tictactoe(Board):
    def __init__(self):
        super().__init__()

    def _check_winner(self):
        # use numpy
        # In current version, winning condition of entire board is searched.
        self.pickup_board = self.state == self.current_player
        if (np.sum(self.pickup_board)) < self.size_of_board:
            return False
        else:
            # horizontal
            for item in self.state.reshape([self.size_of_board, self.size_of_board]):
                if np.sum(item == self.current_player) == self.size_of_board:
                    return True
                
            # vertical
            for item in self.state.reshape([self.size_of_board, self.size_of_board]).T:
                if np.sum(item == self.current_player) == self.size_of_board:
                    return True
                
            # left top to right bottom diagonal
            if np.sum(self.state.reshape([self.size_of_board, self.size_of_board]).diagonal() == self.current_player) == self.size_of_board:
                return True
            
            # right top to left bottom diagonal
            if np.sum(np.fliplr(self.state.reshape([self.size_of_board, self.size_of_board])).diagonal() == self.current_player) == self.size_of_board:
                return True
            
        return False
