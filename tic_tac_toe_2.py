import numpy as np
import random

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_turn = 1

    def available_actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def make_move(self, action):
        self.board[action] = self.current_turn
        self.current_turn = -self.current_turn

    def is_winner(self, player):
        for i in range(3):
            if all(self.board[i, :] == player) or all(self.board[:, i] == player):
                return True
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == player:
            return True
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] == player:
            return True
        return False

    def is_draw(self):
        return not self.is_winner(1) and not self.is_winner(-1) and not any(0 in row for row in self.board)

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_turn = 1

    def print_board(self):
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        for row in self.board:
            print('|'.join(symbols[s] for s in row))
        print('------')

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, decay=0.9995):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay

    def get_state(self, board):
        return str(board.reshape(9))

    def choose_action(self, state, available_actions):
        if np.random.rand() < self.epsilon:
            return random.choice(available_actions)
        q_values = [self.q_table.get((state, action), 0) for action in available_actions]
        max_q_value = max(q_values)
        max_actions = [action for action, q in zip(available_actions, q_values) if q == max_q_value]
        return random.choice(max_actions)

    def update(self, state, action, reward, next_state, done, available_actions):
        max_future_q = max([self.q_table.get((next_state, a), 0) for a in available_actions], default=0)
        current_q = self.q_table.get((state, action), 0)
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q * (not done) - current_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.decay, 0.01)

def train(epochs=10000):
    p1 = QLearningAgent()
    p2 = QLearningAgent()
    game = TicTacToe()

    for _ in range(epochs):
        game.reset()
        done = False

        while not done:
            current_player = p1 if game.current_turn == 1 else p2
            state = current_player.get_state(game.board)
            available_actions = game.available_actions()
            if not available_actions:  # No available actions, the board is full
                break
            
            action = current_player.choose_action(state, available_actions)
            game.make_move(action)

            reward = 0
            next_state = current_player.get_state(game.board)
            done = game.is_winner(game.current_turn) or game.is_draw()

            if game.is_winner(game.current_turn):  # Check if the current player won
                reward = 1
            elif game.is_draw():  # Check if the game is a draw
                reward = 0.5
            
            # Update Q-table for the current player
            current_player.update(state, action, reward, next_state, done, available_actions)
            current_player.decay_epsilon()

    return p1, p2, game

def human_vs_agent(game, agent):
    while True:
        game.print_board()
        if game.current_turn == 1:
            row, col = map(int, input("Enter your move (row col): ").split())
            action = (row, col)
        else:
            state = agent.get_state(game.board)
            action = agent.choose_action(state, game.available_actions())
        
        game.make_move(action)
        if game.is_winner(game.current_turn):
            game.print_board()
            if game.current_turn == 1:
                print("You win!")
            else:
                print("AI wins!")
            break
        elif game.is_draw():
            game.print_board()
            print("It's a draw!")
            break

# Training the agents
p1, p2, game = train()

# Human vs. Agent
game.reset()
human_vs_agent(game, p1)
