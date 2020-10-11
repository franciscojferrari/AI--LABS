import random
random.seed(42)


class TT:
    def __init__(self, unique_fish, indexing):
        self.unique_items = unique_fish + 2  # All unique fish types + the two hooks
        self.size_board = 20
        self.indexing = indexing
        self.hash_table = {}
        self.move_hash_table = {}

    def init_hash_table(self):
        self.hash_table = {}

    def init_move_hash_table(self):
        self.move_hash_table = {}

    def get_table(self):
        return self.hash_table

    def store(
        self, state, alpha, beta, value, depth
    ):
        """Store the current node and it's calculated values in the hash table."""
        self.hash_table[self.compute_hash(state)] = {
            "value": value,
            "depth": depth,
        }

    def store_best_move(self, node, hash, value):
        """Store the best move in hash table.

        Checks the given value against the value stored in the hash table for
        the given node and stores the move leading to that node if value is bigger.
        """
        if hash in self.move_hash_table:
            if node.parent.state.player == 0:
                if value > self.move_hash_table[hash]["value"]:
                    self.move_hash_table[hash] = {"value": value, "move": node.move}
            else:
                if value < self.move_hash_table[hash]["value"]:
                    self.move_hash_table[hash] = {"value": value, "move": node.move}

        else:
            self.move_hash_table[hash] = {"value": value, "move": node.move}

    def compute_hash(self, state):
        """Compute the hash for the given state

        TODO: Pancho how does this work?  Is it correct that the mirrored state is in the same
        dictionary that is converted to the hash?
        """
        hook_positions = state.get_hook_positions()
        fish_positions = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        checking = {}

        for fish_id, position in fish_positions.items():
            x_p, y_p = position

            key_original = str(x_p) + str(y_p)
            checking[key_original] = self.indexing[fish_scores[fish_id]]
            key_mirror = str(abs(x_p - 20)) + str(y_p)
            checking[key_mirror] = self.indexing[fish_scores[fish_id]]

        h0_x, h0_y = hook_positions[0]
        checking[str(h0_x) + str(h0_y)] = self.indexing["h0"]

        h1_x, h1_y = hook_positions[1]
        checking[str(h1_x) + str(h1_y)] = self.indexing["h1"]

        return hash(frozenset(checking.items()))
