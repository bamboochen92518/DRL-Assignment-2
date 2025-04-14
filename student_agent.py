# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
import json
from collections import defaultdict


COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid
        before_add = copy.deepcopy(self.board)

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, before_add

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)


def rot90(pattern):
    return [(y, 3 - x) for x, y in pattern]


def rot180(pattern):
    return [(3 - x, 3 - y) for x, y in pattern]


def rot270(pattern):
    return [(3 - y, x) for x, y in pattern]


def flip_horizontal(pattern):
    return [(3 - x, y) for x, y in pattern]


class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = dict()
        for pattern in self.patterns:
            self.symmetry_patterns[pattern] = self.generate_symmetries(pattern)

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        syms = set()
        for p in [pattern, rot90(pattern), rot180(pattern), rot270(pattern)]:
            # print([hex(4*pts[0] + pts[1]) for pts in p])
            syms.add(tuple(p))
            p = flip_horizontal(p)
            # print([hex(4*pts[0] + pts[1]) for pts in p])
            syms.add(tuple(p))
        return list(syms)

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        return 0 if tile == 0 else int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        total_value = 0
        for pattern, weight_table in zip(self.patterns, self.weights):
            for sym_pattern in self.symmetry_patterns[pattern]:
                feature = self.get_feature(board, sym_pattern)
                total_value += weight_table[feature]
                # print(feature, weight_table[feature])
        # print("Total: ", total_value)
        return total_value

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        # delta /= len(self.patterns)
        for pattern, weight_table in zip(self.patterns, self.weights):
            sym_num = len(self.symmetry_patterns[pattern])
            for sym_pattern in self.symmetry_patterns[pattern]:
                feature = self.get_feature(board, sym_pattern)
                weight_table[feature] += alpha * delta / sym_num

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, state, parent=None, action=None, is_afterstate=False):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node (None for root)
        is_afterstate: True if this node is an afterstate node (chance node), else a regular state (max node)
        prob: probability of reaching this node from its parent (used for chance node)
        """
        self.state = state.copy()
        self.parent = parent
        self.action = action
        self.is_afterstate = is_afterstate

        self.children = {}  # action -> child node
        self.visits = 0
        self.total_reward = 0.0  # accumulated reward for UCB

        # Untried actions or afterstates to expand
        if not is_afterstate:
            env = Game2048Env()
            env.board = state
            self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]
        else:
            self.untried_actions = []  # afterstates will get states as children, not actions

    def fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return len([a for a in range(4) if env.is_move_legal(a)]) == 0


class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, gamma=0.99, V_norm=1.0):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.gamma = gamma
        self.V_norm = V_norm

    def create_env_from_state(self, state, score):
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        if node.is_afterstate:
            # Randomly choose based on tile probabilities
            children = list(node.children.items())
            probs = [child.prob for _, child in children]
            selected = random.choices(children, weights=probs, k=1)[0]
            return selected[1]
        else:
            # UCB selection for max node (player)
            log_parent_visits = math.log(node.visits + 1)
            return max(
                node.children.values(),
                key=lambda child: (child.total_reward / (child.visits + 1e-6)) +
                                  self.c * math.sqrt(log_parent_visits / (child.visits + 1e-6))
            )

    def evaluate(self, env):
        # If node is a max node (state), evaluate all its afterstates
        best_value = 0
        for action in range(4):
            if env.is_move_legal(action):
                sim_env = copy.deepcopy(env)
                _, reward, done, after_state = sim_env.step(action)
                value = reward + self.approximator.value(after_state)
                best_value = max(best_value, value)
        return best_value

    def backpropagate(self, node, norm_value):
        while node is not None:
            node.visits += 1
            node.total_reward += norm_value
            node = node.parent

    def expand(self, node, sim_env):
        # Expand all legal player moves (state -> afterstate)
        if not node.is_afterstate:
            legal_move = node.untried_actions
            action = legal_move.pop()
            _, reward, done, after_state = sim_env.step(action)
            sim_env.board = after_state
            after_node = TD_MCTS_Node(
                state=after_state,
                parent=node,
                action=action,
                is_afterstate=True
            )
            after_node.value = self.approximator.value(after_state)
            node.children[action] = after_node

            node = after_node

        # Expand all possible tile spawns (afterstates -> new states)
        empty_cells = [(i, j) for i in range(4) for j in range(4) if node.state[i][j] == 0]
        if not empty_cells:
            return node
        lucky_cell = random.choice(empty_cells)
        i, j = lucky_cell

        if random.random() < 0.9:
            sim_env.board[i][j] = 2
        else:
            sim_env.board[i][j] = 4

        if (i, j, sim_env.board[i][j]) not in node.children.keys():
            child_node = TD_MCTS_Node(
                state=sim_env.board,
                parent=node,
                action=None,
                is_afterstate=False,
            )
            node.children[(i, j, sim_env.board[i][j])] = child_node

        return node.children[(i, j, sim_env.board[i][j])]

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, 0)

        # --- SELECTION ---
        while not node.untried_actions and node.children:
            assert np.array_equal(sim_env.board, node.state)
            """

            if node.is_afterstate:
                print(node.children)
                print(node.state)
            """
            log_parent_visits = math.log(node.visits + 1)
            best_node = max(node.children.values(), key=lambda child: (child.total_reward / (child.visits + 1e-6)) + self.c * math.sqrt(log_parent_visits / (child.visits + 1e-6)))
            _, _, _, afterstate = sim_env.step(best_node.action)
            sim_env.board = afterstate
            node = best_node
            node = self.expand(node, sim_env)

        # --- EXPANSION ---
        if node.untried_actions:
            node = self.expand(node, sim_env)

        # --- EVALUATION ---
        value = self.evaluate(sim_env)
        norm_value = value / self.V_norm

        # --- BACKPROPAGATION ---
        self.backpropagate(node, norm_value)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution


def f2t(f: str):
    hex_number = hex(int(f))
    hex_number = hex_number[2:]
    hex_number = "0" * (5 - len(hex_number)) + hex_number
    reversed_hex = hex_number[::-1]
    decimal_values = []
    for char in reversed_hex:
        decimal_value = int(char, 16)
        decimal_values.append(decimal_value)

    return tuple(decimal_values)


def reorder_action(a: list):
    ret = []
    order = [0, 3, 1, 2]
    for o in order:
        if o in a:
            ret.append(o)
    return ret


patterns = [
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),          # T01259
        ((1, 2), (2, 2), (2, 3), (3, 0), (3, 1)),          # dog 6, 10, 11, 12, 13
        ((0, 1), (0, 2), (0, 3), (1, 1), (2, 1)),          # L
    ]

file_path = "approximator-61300000.json"
with open(file_path, "r") as file:
    weight_table = json.load(file)
approximator = NTupleApproximator(board_size=4, patterns=patterns)

env = Game2048Env()
td_mcts = TD_MCTS(env, approximator, iterations=100, exploration_constant=1.41, gamma=1, V_norm=10000)

i = 0
for fn, wt in weight_table.items():
    for f, w in wt.items():
        t = f2t(f)
        approximator.weights[i][t] = w
        # print(t, w)
    i += 1


def get_action(state, score):
    root = TD_MCTS_Node(state)

    # Run multiple simulations to build the MCTS tree
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    # Select the best action (based on highest visit count)
    best_act, _ = td_mcts.best_action_distribution(root)

    return best_act
