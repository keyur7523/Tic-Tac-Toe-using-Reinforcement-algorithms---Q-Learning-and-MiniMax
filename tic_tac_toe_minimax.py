import numpy as np
import random
from collections import defaultdict

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_turn = 1  # Player 1 starts

    def available_actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == 0]

    def make_move(self, action):
        if self.board[action] == 0:
            self.board[action] = self.current_turn
            self.current_turn = -1 if self.current_turn == 1 else 1
            return True
        return False

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
        return all(self.board[i][j] != 0 for i in range(3) for j in range(3))

    def reset_board(self):
        self.board = np.zeros((3, 3), dtype=int)

    def get_state(self):
        return str(self.board.reshape(9))
                     
    def human_move(self):
        while True:
            try:
                ip = str(input("Enter row and col (0, 1, or 2): "))
                row, col = int(ip[0]), int(ip[1])
                #col = int(input("Enter column (0, 1, or 2): "))
                if (row in [0, 1, 2], col in [0, 1, 2]) and self.board[row][col] == 0:
                    self.board[row][col] = self.current_turn
                    self.current_turn = -1 if self.current_turn == 1 else 1
                    break
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Please enter a valid row and column.")

    def print_board(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        for row in self.board:
            print('|'.join(symbols[i] for i in row))
            print("-" * 5)
        print("-" * 5)
                   
class MinimaxAgent:
    def __init__(self):
        self.player_symbol = -1

    def minimax(self, board, depth, is_maximizing):
        if check_winner(board, self.player_symbol):
            return 10 - depth
        elif check_winner(board, -self.player_symbol):
            return depth - 10
        elif is_board_full(board):
            return 0

        if is_maximizing:
            best_score = -np.inf
            for action in get_available_actions(board):
                board[action] = self.player_symbol
                score = self.minimax(board, depth + 1, False)
                board[action] = 0  # Undo move
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = np.inf
            for action in get_available_actions(board):
                board[action] = -self.player_symbol
                score = self.minimax(board, depth + 1, True)
                board[action] = 0  # Undo move
                best_score = min(score, best_score)
            return best_score

    def choose_action(self, board):
        best_score = -np.inf
        best_move = None
        for action in get_available_actions(board):
            board[action] = self.player_symbol
            score = self.minimax(board, 0, False)
            board[action] = 0  # Undo move
            if score > best_score:
                best_score = score
                best_move = action
        return best_move

# Utility functions
def check_winner(board, player):
    # Check rows, columns, and diagonals for a win
    for i in range(3):
        if all(board[i, :] == player) or all(board[:, i] == player):
            return True
    if board[0, 0] == board[1, 1] == board[2, 2] == player:
        return True
    if board[0, 2] == board[1, 1] == board[2, 0] == player:
        return True
    return False

def is_board_full(board):
    return all(board[i][j] != 0 for i in range(3) for j in range(3))

def get_available_actions(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]

def play_against_minimax(game, minimax_agent, human_player=1):
    while not game.is_winner(-1) and not game.is_winner(1) and not game.is_draw():
        game.print_board()
        if game.current_turn == human_player:
            print("\nYour turn (Human):")
            game.human_move()
        else:
            print("\nMinimax Agent's turn:")
            board_copy = np.copy(game.board)  # Create a copy of the board
            action = minimax_agent.choose_action(board_copy)
            game.make_move(action)

        if game.is_winner(game.current_turn):
            winner = "Human" if game.current_turn == human_player else "Minimax Agent"
            print(f"{winner} wins!")
            break
        elif game.is_draw():
            print("It's a draw!")
            break

    print("The game has ended.")
    game.print_board()
    
# Function to play against the trained Q-agent
def play_against_agent(agent):
    game = TicTacToe()
    human_player = 1  # Human is player 1, agent is player -1

    print("The player has entered the game")
    while not game.is_winner(-1) and not game.is_winner(1) and not game.is_draw():
        if game.current_turn == human_player:
            print("\nYour turn (Human):")
            game.print_board()
            game.human_move()
        else:
            print("\nAgent's turn:")
            state = game.get_state()
            action = agent.best_action(state, game.available_actions())
            game.make_move(action)
            game.print_board()

        if game.is_winner(game.current_turn):
            winner = "Human" if game.current_turn == human_player else "Agent"
            print(f"{winner} wins!")
            break
        elif game.is_draw():
            print("It's a draw!")
            break
        
    print("The game board will be played.")
    game.print_board()

class QAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.9):
        self.q_table = {}
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01

    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)  # Explore
        else:
            return self.best_action(state, available_actions)  # Exploit

    def best_action(self, state, available_actions):
        q_values, q_v = [], []
        for action in available_actions:
            q_values.append(self.get_q_value(state, action))
            q_v.append((action, self.get_q_value(state, action)))
        
        #q_values = [self.get_q_value(state, action) for action in available_actions]
        # print(f"q_values : {q_v}\nstate : {state}")
        max_q = max(q_values)
        # In case there are several actions with the same Q-value   
        actions_with_max_q = [action for action, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(actions_with_max_q)

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0)

    def update_q_table(self, state, action, reward, next_state, next_available_actions):
        current_q = self.get_q_value(state, action)
        max_future_q = max([self.get_q_value(next_state, next_action) for next_action in next_available_actions], default=0)
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        self.q_table[(state, action)] = new_q
        
        #for key, value in self.q_table.items():
        #    print(f"key : {key} value : {value}")
           
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
def play_against_agent(agent):
    game = TicTacToe()
    human_player = 1  # Human is player 1, agent is player -1

    while not game.is_winner(-1) and not game.is_winner(1) and not game.is_draw():
        if game.current_turn == human_player:
            print("\nYour turn (Human):")
            game.print_board()
            game.human_move()
        else:
            print("\nAgent's turn:")
            state = game.get_state()
            action = agent.best_action(state, game.available_actions())
            game.make_move(action)
            game.print_board()

        if game.is_winner(game.current_turn):
            winner = "Human" if game.current_turn == human_player else "Agent"
            print(f"{winner} wins!")
            break
        elif game.is_draw():
            print("It's a draw!")
            break

    print("The game has ended.")
    game.print_board()

def train_agent(episodes):
    player1 = QAgent()
    player2 = QAgent()
    game = TicTacToe()
    training_agent_1, training_agent_2, scaling_factor = 0, 0, 1
    training_scale = []

    for episode in range(episodes):
        game.reset_board()
        state_2, action_2, new_state_2 = None, None, None  # Initialize state and action for player 2

        while True:
            # Player 1's turn
            state_1 = game.get_state()
            available_actions = game.available_actions()
            action_1 = player1.choose_action(state_1, available_actions)
            game.make_move(action_1)
            new_state_1 = game.get_state()

            reward = 0
            if game.is_winner(1):  # Player 1 wins
                reward = 1
                training_agent_1 += 1
                if episode % scaling_factor == 0:
                    training_scale.append((training_agent_1, training_agent_2))
            elif game.is_draw():  # Draw
                reward = 0.5

            player1.update_q_table(state_1, action_1, reward, new_state_1, game.available_actions())

            if reward != 0:  # Game has ended
                player2.update_q_table(state_2, action_2, -reward, new_state_2, game.available_actions())  # Opposite reward for the other player
                break

            # Player 2's turn
            state_2 = game.get_state()
            action_2 = player2.choose_action(state_2, game.available_actions())
            game.make_move(action_2)
            new_state_2 = game.get_state()

            reward = 0
            if game.is_winner(-1):  # Player 2 wins
                reward = 1
                training_agent_2 += 1
                if episode % scaling_factor == 0:
                    training_scale.append((training_agent_1, training_agent_2))
            elif game.is_draw():  # Draw
                reward = 0.5

            player2.update_q_table(state_2, action_2, reward, new_state_2, game.available_actions())

            if reward != 0:  # Game has ended
                player1.update_q_table(state_1, action_1, -reward, new_state_1, game.available_actions())  # Opposite reward for the other player
                break

            # Decay epsilon after each move
            player1.update_epsilon()
            player2.update_epsilon()

    return player1, player2, training_scale

def play_game(q_agent, minimax_agent, games):
    count_dict = defaultdict(int)
    count_minimax, count_qagent, count_draw = 0, 0, 0
    count_list = [(0, 0, 0)]
    for g in range(1, games+1):    
        flag_draw = 0
        game = TicTacToe()
        while not game.is_winner(-1) and not game.is_winner(1) and not game.is_draw():
            if game.current_turn == 1:
                # Player 1 (Q-learning agent) makes a move
                state = game.get_state()
                action = q_agent.best_action(state, game.available_actions())
            elif game.current_turn == -1:
                # Player 2 (Minimax agent) makes a move
                board_copy = np.copy(game.board)
                action = minimax_agent.choose_action(board_copy)

            game.make_move(action)
            #game.print_board()

            #print(f"game.current_turn {game.current_turn} game.is_winner(game.current_turn) {game.is_winner(game.current_turn)}")

            if game.is_winner(game.current_turn):
                winner = "Q-Agent" if game.current_turn == 1 else "Minimax Agent"
                print(f"{winner} wins!")
                break
            elif game.is_draw():
                print("It's a draw!")
                flag_draw = 1
                break
        if flag_draw: 
            count_dict['draw'] += 1
            count_draw += 1
        else:
            if game.current_turn: 
                count_dict['Minimax Agent'] += 1
                count_minimax += 1
            elif game.current_turn: 
                count_dict['Q-Agent'] += 1
                count_qagent += 1
                
        if g % 10 == 0:
            count_list.append((count_qagent, count_minimax, count_draw))

        print("The game has ended.")
    
    return count_dict, count_list

#minimax_agent = MinimaxAgent()
#game = TicTacToe()
#play_against_minimax(game, minimax_agent)

# Main execution
player1, player2, training_scale = train_agent(episodes=80000)  # Train the Q-learning agent

while 1:
    play_against_agent(player2)
    
while 1:
    game = TicTacToe()
   
    play_against_minimax()

#count_dict, count_list = play_game(player1, minimax_agent, games = 100)
#rint(f"count_dict : {count_dict}")
#print(f"count_list : {count_list}")