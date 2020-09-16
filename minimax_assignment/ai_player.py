import math
import random


class MinMaxModel(object):
    def __init__(self, depth):
        self.depth = depth
        self.boundary_location = 20
        self.fish_values = None

    def init_fish_values(self, init_data):
        init_data.pop("game_over")
        updated_fish_values = {}
        for fish, value in init_data.items():
            updated_fish_values[int(fish[-1])] = value["score"]

        self.fish_values = updated_fish_values

    def simple_heuristic(self, state):
        # scores =
        max_score, min_score = state.get_player_scores()
        return max_score - min_score

    def best_next_move(self, node):
        possible_values = {}
        alpha = -math.inf
        for child in node.compute_and_get_children():
            mini_max_result = self.minimax_algorith(child, self.depth, alpha)
            if mini_max_result >= alpha:
                alpha = mini_max_result
                possible_values[child.move] = mini_max_result
        # if all(value == possible_values[0] for value in possible_values.values()):
        #     return random.randint(0, 4)
        return max(possible_values, key=possible_values.get)

    def minimax_algorith(self, node, depth, alpha=-math.inf, betta=math.inf):
        if depth == 0 or len(node.compute_and_get_children()) == 0:
            # Calculate Heuristics
            return self.get_heuristic(node)
        if node.state.player == 0:
            maxEval = -math.inf
            for child in node.compute_and_get_children():
                maxEval = max(
                    maxEval, self.minimax_algorith(child, depth - 1, alpha, betta)
                )
                alpha = max(alpha, maxEval)
                if alpha >= betta:
                    # Alpha betta prunning. We stop the interation on the children nodes
                    break
            return maxEval
        else:
            minEval = math.inf
            for child in node.compute_and_get_children():
                minEval = min(
                    minEval, self.minimax_algorith(child, depth - 1, alpha, betta)
                )
                betta = min(betta, minEval)
                if betta <= alpha:
                    # Alpha betta prunning. We stop the interation on the children nodes
                    # Alpha cut-off
                    break
            return minEval

    def get_heuristic(self, node):
        """
        - Distance of hook to closest fish
        - Include score of the fish
        - Include score heuristic
        :return:
        """
        distance_heuristic = self.get_distance_heuristic(node.state)
        score_heuristic = self.get_score_heuristic(node.state)

        return distance_heuristic + score_heuristic

    def get_distance_heuristic(self, state):
        hook_positions = state.get_hook_positions()
        fish_positions = state.get_fish_positions()
        fish_scores = state.fish_scores
        scores = []  # heuristics for both players

        for player, hook_position in hook_positions.items():
            heuristic = 0
            for fish_id, location in fish_positions.items():

                y_difference = abs(hook_position[0] - location[0])
                # get the minimum score of the difference in x
                x_difference = min(
                    abs(hook_position[1] - location[1]),
                    abs(hook_position[1] - self.boundary_location - location[1]),
                )

                distance = math.sqrt(x_difference ** 2 + y_difference ** 2)
                heuristic += fish_scores[fish_id] * (1 / (distance + 1e-4))
            scores.append(heuristic)
        return scores[0] - scores[1]

    def get_score_heuristic(self, state):
        max_score, min_score = state.get_player_scores()
        fish_caught_max, fish_caught_min = state.get_caught()
        fish_score_max = (
            self.fish_values[fish_caught_max] if fish_caught_max != None else 0
        )
        fish_score_min = (
            self.fish_values[fish_caught_min] if fish_caught_min != None else 0
        )
        return max_score - min_score + fish_score_max - fish_score_min
