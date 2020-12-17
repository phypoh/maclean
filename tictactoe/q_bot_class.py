import numpy as np
import random
from board_rules import *

debug_mode = False


class Bot():

    def __init__(self, epsilon, board_size, alpha=0.9, gamma=0.99):

        self.epsilon = epsilon
        self.library = {}
        self.to_update_X = []
        self.to_update_O = []
        self.board_size = board_size
        self.alpha = alpha
        self.gamma = gamma

    def init_board(self, length):
        """Initialise an empty board"""
        board = np.zeros((length, length), dtype=int)
        return board

    def decide_move(self, board, side, training_mode=False):
        if not training_mode: print("Bot's move.")
        if board.tobytes() not in self.library:
            if debug_mode and not training_mode: print("Unseen board situation")
            moves = self.init_state(board)
            selected = random.choice(moves)
            reward = 0.5

        else:
            roll = random.uniform(0, 1)
            moves = self.check_moves(board)
            if roll > self.epsilon*training_mode:
                # greedy mode
                if debug_mode and not training_mode: print("Greedy mode")
                selected, reward = self.max_reward(board)

            else:
                # explore mode
                if debug_mode and not training_mode: print("Explore mode")
                selected = random.choice(moves)
                reward = self.library[board.tobytes()][selected]['Reward']

        if debug_mode and not training_mode: print(self.library[board.tobytes()], selected)
        if side == 1:
            self.to_update_X.append([board.copy(), selected])
        elif side == -1:
            self.to_update_O.append([board.copy(), selected])

        # print(self.to_update_X)
        # print(self.to_update_O)

        selected -= 1
        position = (int(selected/3) + 1, selected%3 + 1)

        if not training_mode: print("Bot puts piece at", position)
        if not training_mode: print("Expected reward:", reward)

        return position


    def check_moves(self, board):
        board_flat = board.flatten()
        moves = [i for i,x in enumerate(board_flat) if x == 0]
        moves = [i + 1 for i in moves]
        return moves


    def init_state(self, board):
        self.library[board.tobytes()] = {}
        moves = self.check_moves(board)
        for move in moves:
            self.library[board.tobytes()][move] = {'Reward': 0.5, 'Trials': 0}
        return moves


    def max_reward(self, board):
        reward = 0
        selected = None
        for move in self.library[board.tobytes()]:
            if self.library[board.tobytes()][move]['Reward'] >= reward:
                reward = self.library[board.tobytes()][move]['Reward']
                selected = move
        return selected, reward

    def update_bot(self, win):
        if win == 1:
            for row in self.to_update_X:
                board = row[0]
                selected = row[1]
                self.update_state(board, selected, 1)

            for row in self.to_update_O:
                board = row[0]
                selected = row[1]
                self.update_state(board, selected, 0)

        elif win == -1:
            for row in self.to_update_O:
                board = row[0]
                selected = row[1]
                self.update_state(board, selected, 1)

            for row in self.to_update_X:
                board = row[0]
                selected = row[1]
                self.update_state(board, selected, 0)

        elif win == 2:
            for row in self.to_update_X:
                board = row[0]
                selected = row[1]
                self.update_state(board, selected, 0.5)

            for row in self.to_update_O:
                board = row[0]
                selected = row[1]
                self.update_state(board, selected, 0.5)

        self.to_update_X = []
        self.to_update_O = []

    def update_state(self, board, selected, reward):
        self.library[board.tobytes()][selected]['Trials'] += 1
        x, max_rew = self.max_reward(board)
        self.library[board.tobytes()][selected]['Reward'] += self.alpha * (reward + self.gamma * max_rew - self.library[board.tobytes()][selected]['Reward'] )


    def train(self, iterations):
        print("Training bot...")
        for i in range(iterations):
            if debug_mode: print("Iteration number", i+1)

            board = self.init_board(self.board_size)
            win = 0
            side = 1

            while win == 0:
                position = self.decide_move(board, side, training_mode = True)
                board, err_check = place_piece(board, position, side)
                side = -side
                win = check_win(board)


            if debug_mode: print("Rewarding bot...")
            self.update_bot(win)
            if debug_mode: print("Rewarded.")

        print("Trained.")

