import math
import random
import time

from fishing_game_core.shared import ACTION_TO_STR
from tranposition_table import TT


class MiniMaxModel(object):
    def __init__(self, depth, init_data):
        self.depth = depth
        self.boundary_location = 20
        self.fish_values = None
        self.init_fish_values(init_data)
        self.iterative_deepening_depth = 7
        self.timeout_s = 58e-3
        self.start_time = None
        self.timeout = False
        self.TT = TT(len(self.fish_values), self.indexing)
        self.best_moves_count = {}
        self.move_order_method = "best_move_first"  # Can be "none", "random", "move_count", "best_move_first"

    def init_fish_values(self, init_data):
        init_data.pop("game_over")
        updated_fish_values = {}
        indexing = {}
        i = 0
        for fish, value in init_data.items():
            updated_fish_values[int(fish[-1])] = value["score"]
            if value["score"] not in indexing:
                indexing[value["score"]] = i
                i+=1
        self.fish_values = {int(fish.split('fish')[1]):value['score'] for fish, value in init_data.items()}
        indexing['h0'] = i
        indexing['h1'] = i+1
        self.indexing = indexing

    def minimax_algorithm(
        self, node, depth, alpha=-math.inf, beta=math.inf
    ):
        """Recursive implementation of the minimax algorithm with alpha-beta pruning."""
        if depth == 0:
            return self.get_heuristic(node)

        hash = self.TT.compute_hash(node.state)
        hash_table = self.TT.get_table()

        #  Check if state has been visited before.
        if hash in hash_table:
            tt_entry = hash_table[hash]

            # Check if the depth of the stored state is deeper than current state.
            if depth <= tt_entry["depth"]:  # TODO  check if  this needs to be smaller
                return tt_entry["value"]

        # Run minimax for the max player.
        if node.state.player == 0:
            # Check if almost out of time, if so, return beta value to  parent node.
            if time.time() - self.start_time > self.timeout_s:
                self.timeout = True
                return alpha

            children = self.get_children(node)
            if len(children) == 0:
                return self.get_heuristic(node)

            value = -math.inf
            for child in children:

                if self.move_not_possible(child):
                    continue

                # Recursive call.
                value = max(
                    value, self.minimax_algorithm(child, depth - 1, alpha, beta)
                )
                alpha = max(alpha, value)

                # Pruning condition, stop loop over children.
                if alpha >= beta or value == math.inf:
                    break
                self.TT.store_best_move(child, hash, value)

            self.TT.store(node.state, alpha, beta, value, depth)

            return value

        else:
            # Check if almost out of time, if so, return beta value to  parent node.
            if time.time() - self.start_time > self.timeout_s:
                self.timeout = True
                return beta

            children = self.get_children(node)
            if len(children) == 0:
                return self.get_heuristic(node)

            value = math.inf
            for child in children:
                if self.move_not_possible(child):
                    continue

                # Recursive call
                value = min(
                    value, self.minimax_algorithm(child, depth - 1, alpha, beta)
                )
                beta = min(beta, value)

                # Pruning condition, stop loop over children.
                if beta <= alpha or value == -math.inf:
                    break
                self.TT.store_best_move(child, hash, value)

            self.TT.store(node.state, alpha, beta, value, depth)

            return value

    def get_heuristic(self, node):
        """Calculate heuristic for current node.

        Heuristic consists of 2 parts. A heuristic based on the score of the players and a heuristic
        based on the proximity to the fish weighted by their score.

        Returns:
            heuristic: Score
        """
        max_score, min_score = node.state.get_player_scores()
        fish_caught_max, fish_caught_min = node.state.get_caught()

        fish_positions = node.state.get_fish_positions()
        hook_positions = node.state.get_hook_positions()

        score_heuristic = self.get_score_heuristic(
            fish_caught_max, fish_caught_min, max_score, min_score
        )

        number_of_fish = len(fish_positions)

        #  if there are no more fish left in the tank, return the maximum score depending on the player
        if number_of_fish == 0 or number_of_fish == len([fish_caught_min,  fish_caught_max]):
            if score_heuristic > 0:
                return math.inf
            if score_heuristic < 0:
                return -math.inf
            return 0.0

        distance_heuristic = self.get_distance_heuristic(
            hook_positions, fish_positions, fish_caught_max, fish_caught_min
        )
        return distance_heuristic + score_heuristic * 5

    def get_distance_heuristic(
        self,
        hook_positions,
        fish_positions,
        fish_caught_max,
        fish_caught_min,
    ):
        """Calculate distance heuristic.

        The distance heuristic is calculated by calculating the distance to every fish for both
        players weighted by the score of the fish. The score for the max  player minus the score
        for the min player is returned as the distance heuristic.

        Returns:
            Distance heuristic.
        """
        scores = []  # heuristics for both players

        for player, hook_position in hook_positions.items():
            heuristic = 0
            for fish_id, fish_location in fish_positions.items():
                if fish_id in (fish_caught_max, fish_caught_min):
                    continue
                else:
                    y_difference = abs(fish_location[1] - hook_position[1])
                    x_difference = min(
                        abs(fish_location[0] - hook_position[0]),
                        self.boundary_location
                        - abs(fish_location[0] - hook_position[0]),
                    )
                    distance = x_difference + y_difference
                heuristic += self.fish_values[fish_id] * math.exp(-1 * distance)
            scores.append(heuristic)
        return scores[0] - scores[1]

    def get_score_heuristic(
        self,
        fish_caught_max,
        fish_caught_min,
        max_score,
        min_score,
    ):
        """Calculate score heuristic.

        The score heuristic is calculated by adding the score for each player in that state with
        the fish that is currently on  the hook. The score for the max  player minus the score
        for the min player is returned as the score heuristic.

        Returns:
            Score heuristic.
        """
        # fish_score_max = self.fish_values[fish_caught_max] if fish_caught_max else 0
        # fish_score_min = self.fish_values[fish_caught_min] if fish_caught_min else 0

        fish_score_max = (
            self.fish_values[fish_caught_max] if fish_caught_max != None else 0
        )

        fish_score_min = (
            self.fish_values[fish_caught_min] if fish_caught_min != None else 0
        )

        return max_score - 0.5*min_score + fish_score_max - 0.5*fish_score_min



    def search_best_next_move_iterative_deepening(self, initial_tree_node):
        """Iterative deepening search.

        The iterative deepening search (IDS) is a method that iteratively increases the depth at
        which it is searching. Before every search the hash tables are initialized. In these hash
        tables the visited nodes are stored such that they don't need to be visited again. When
        looping over the children of the initial tree node, the optimal move is found by comparing
        the heuristic that is returned by  the minimax algorithm. During the search several time
        checks are implemented to prevent a time out.

        Returns:
            Optimal action found by iterative deepening search.
        """
        self.start_time = time.time()
        self.global_best_move = random.randint(0, 4)
        self.global_best_alpha = -math.inf
        self.timeout = False

        self.TT.init_hash_table()
        self.TT.init_move_hash_table()

        for depth in range(2, self.iterative_deepening_depth):

            children = self.get_children(initial_tree_node)
            best_move = random.randint(0, 4)
            alpha = -math.inf

            for child in children:
                if self.move_not_possible(child):
                    continue
                if self.timeout:
                    self.save_best_move(self.global_best_move)
                    return ACTION_TO_STR[self.global_best_move]
                value = self.minimax_algorithm(child, depth, -math.inf, math.inf)

                if value >= alpha:
                    alpha = value
                    best_move = child.move

                if self.timeout:
                    self.save_best_move(self.global_best_move)
                    return ACTION_TO_STR[self.global_best_move]

            self.global_best_move = best_move

        self.save_best_move(self.global_best_move)
        return ACTION_TO_STR[self.global_best_move]

    def save_best_move(self, move):
        """Saves the move by incrementing the move count."""
        if move in self.best_moves_count:
            self.best_moves_count[move] += 1
        else:
            self.best_moves_count[move] = 1

    def get_children(self, node):
        """Method that returns children of given node based on set move order method.

        Various move order methods:
            Move Count: Count the number of moves previously made and start with evaluation
                that move first.
            Random: Shuffle the list of children.
            best Move First: Store which move was best leading to the parent node and use
                that move first.

        Returns:
            List of ordered children.
        """
        children = node.compute_and_get_children()
        if len(children) == 1:
            return children
        elif self.move_order_method == "move_count":
            order = [
                k
                for k, v in sorted(
                    self.best_moves_count.items(), key=lambda item: item[1]
                )
            ]
            children_ordered = [None for _ in order]

            for child in children:
                if child.move in order:
                    children_ordered[order.index(child.move)] = child
                else:
                    children_ordered.append(child)

            return children_ordered

        elif self.move_order_method == "random":
            random.shuffle(children)
            return children
        elif self.move_order_method == "best_move_first":
            parent_hash = self.TT.compute_hash(node.state)
            if parent_hash in self.TT.move_hash_table:
                best_move = children.pop(self.TT.move_hash_table[parent_hash]["move"])
                random.shuffle(children)
                children.insert(0, best_move)
                return children
            else:
                random.shuffle(children)
                return children

        else:
            return children

    @staticmethod
    def move_not_possible(node):
        """Check if the move is possible."""
        if node.move not in [3, 4]:
            return False

        hook_positions = node.state.get_hook_positions()

        diff = hook_positions[0][0] - hook_positions[1][0]  # player max - player min

        if abs(diff) == 1:
            if node.state.player == 0:
                if diff == 1:
                    if node.move == 4:
                        return True
                elif diff == -1:
                    if node.move == 3:
                        return True

            else:
                if diff == 1:
                    if node.move == 3:
                        return True
                elif diff == -1:
                    if node.move == 4:
                        return True

        return False
