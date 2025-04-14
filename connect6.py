import sys
import numpy as np
import random
import copy
import math
from loguru import logger


logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")


class Node:
    def __init__(self, game, move, parent=None):
        self.game = game  # Game state
        self.move = move  # Move that led to this state (r, c, color)
        self.parent = parent  # Parent node
        self.children = {}  # List of child nodes
        self.visits = 0  # Number of visits to this node
        self.total_rewards = 0
        self.untried_moves = self.get_legal_moves()  # Possible moves to try

    def get_legal_moves(self):
        """Returns all legal moves for the current game state."""
        empty_positions = [(r, c) for r in range(self.game.size) for c in range(self.game.size) if self.game.board[r, c] == 0]
        # random.shuffle(empty_positions)
        return empty_positions

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_moves) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, game, iterations=50, exploration_constant=1.41, rollout_depth=0, gamma=1):
        self.game = game
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_game_from_board(self, board):
        # Create a deep copy of the environment with the given state and score.
        new_game = copy.deepcopy(self.game)
        new_game.board = board.copy()
        return new_game

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        best_value = float('-inf')
        best_child = None
        for child in node.children.values():
            uct_value = (child.total_rewards / (child.visits + 1e-6)) + self.c * math.sqrt(math.log(node.visits + 1) / (child.visits + 1e-6))
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
        return best_child

    def rollout(self, sim_game, depth, node):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        for _ in range(depth):
            color = sim_game.whose_turn()
            empty_positions = [(r, c) for r in range(self.game.size) for c in range(self.game.size) if sim_game.board[r, c] == 0]
            if not empty_positions:
                break
            r, c = random.choice(empty_positions)
            sim_game.board[r, c] = color
            """
            if sim_game.check_win():
                score = {1: 0, 2: 0}
                score[sim_game.check_win()] = 15000
                return score
            """
        r, c, color = node.move
        op_color = 3 - color
        constant = 0.5 if sim_game.evaluate_board()[op_color] < 1000 else 10
        a = sim_game.evaluate_position(r, c, color)
        b = sim_game.evaluate_position(r, c, op_color)
        logger.debug(f'm = {a}, o = {b}, r, c = {r}, {c}, color = {color}')
        sim_game.board[r][c] = color
        return a + b

    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        while node is not None:
            node.visits += 1
            node.total_rewards += reward
            reward *= self.gamma  # Apply discount factor
            node = node.parent

    def run_simulation(self, root, my_color):
        node = root
        sim_game = self.create_game_from_board(node.game.board)
        my_color = 1 if my_color == 'B' else 2

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            r, c, my_color = node.move
            sim_game.board[r][c] = my_color

        # TODO: Expansion: If the node is not terminal, expand an untried action.
        if node.untried_moves:
            action = node.untried_moves.pop()
            r, c = action
            turn = sim_game.whose_turn()
            sim_game.board[r][c] = turn
            new_node = Node(copy.deepcopy(sim_game), (r, c, turn), parent=node)
            node.children[(r, c, turn)] = new_node
            node = new_node

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_game, self.rollout_depth, node)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = {}
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.total_rewards > best_visits:
                best_visits = child.total_rewards
                best_action = action
        return best_action, distribution


class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False
        self.last_opponent_move = None
        self.MCT = TD_MCTS(self)

    def reset_board(self):
        """Resets the board and game state."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Changes board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        """Checks if a player has won. Returns 1 (Black wins), 2 (White wins), or 0 (no winner)."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts a column index to a letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))

    def label_to_index(self, col_char):
        """Converts a column letter to an index (handling the missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Processes a move and updates the board."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            col_char = stone[0].upper()
            col = self.label_to_index(col_char)
            row = int(stone[1:]) - 1
            if not (0 <= row < self.size and 0 <= col < self.size) or self.board[row, col] != 0:
                print("? Invalid move")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.last_opponent_move = positions[-1]  # Track the opponent's last move
        self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    def mask(self, empty_positions):
        points = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] != 0]
        if not points:
            return empty_positions
        min_x = max_x = points[0][0]
        min_y = max_y = points[0][1]

        # Find the min and max coordinates
        for p in points:
            x, y = p[0], p[1]
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

        ret = []

        for ep in empty_positions:
            if min_x - 2 <= ep[0] <= max_x + 2 and min_y - 2 <= ep[1] <= max_y + 2:
                ret.append(ep)

        return ret

    @logger.catch
    def generate_move(self, color):
        """Generates the best move based on predefined rules and ensures output."""

        if self.game_over:
            print("? Game over", flush=True)
            return

        root = Node(game=copy.deepcopy(game), move=None)

        logger.debug('START SIMULATE')
        for _ in range(len(root.untried_moves)):
            self.MCT.run_simulation(root, my_color=color)

        best_action, _ = self.MCT.best_action_distribution(root)

        move_str = f"{self.index_to_label(best_action[1])}{best_action[0] + 1}"
        self.play_move(color, move_str)
        print(move_str, flush=True)
        return

    def evaluate_board(self):
        """Checks if a player has won. Returns 1 (Black wins), 2 (White wins), or 0 (no winner)."""
        # logger.debug(self.board)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        score = {1: 0, 2: 0}
        cs = {0: 0, 1: 1, 2: 10, 3: 100, 4: 1000, 5: 10000, 6: 100000}
        for r in range(self.size):
            for c in range(self.size):
                for dr, dc in directions:
                    counter = [0, 0, 0]
                    rr, cc = r, c
                    for i in range(6):
                        if 0 <= rr < self.size and 0 <= cc < self.size:
                            counter[self.board[rr, cc]] += 1
                            rr += dr
                            cc += dc
                        else:
                            break
                    if sum(counter) == 6:
                        if counter[1] == 0:
                            score[2] += cs[counter[2]]
                        if counter[2] == 0:
                            score[1] += cs[counter[1]]

        return score

    def evaluate_position(self, r, c, color):
        """Evaluates the strength of a position based on alignment potential."""
        self.board[r, c] = color
        score = self.evaluate_board()
        self.board[r, c] = 0
        return score[color]

    def show_board(self):
        """Displays the board in text format."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row + 1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print("env_board_size=19", flush=True)

        if not command:
            return

        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

    def whose_turn(self):
        black_num = np.count_nonzero(self.board == 1)
        white_num = np.count_nonzero(self.board == 2)

        if black_num % 2 == 1 and black_num >= white_num:
            return 2
        if white_num % 2 == 0 and white_num >= black_num:
            return 1
        print('whose_turn is wrong')


if __name__ == "__main__":
    game = Connect6Game()
    game.run()
