import math
import random
import multiprocessing.dummy as mp 


class MinMaxModel(object):
    def __init__(self, depth):
        self.depth = depth

    def simple_heuristic(self, node):
        max_score, min_score = node.state.get_player_scores()
        return max_score - min_score

    def best_next_move(self, node):
        possible_values = {}
        alpha = -math.inf
        for child in node.compute_and_get_children():
            mini_max_result = self.minimax_algorith(child, self.depth, alpha)
            if mini_max_result >= alpha:
                alpha = mini_max_result
                possible_values[child.move] = mini_max_result
        if all(value == possible_values[0] for value in possible_values.values()):
            return random.randint(0, 4)
        return max(possible_values, key=possible_values.get)

    def minimax_algorith(self, node, depth, alpha = -math.inf, betta = math.inf):
        if(depth == 0 or len(node.compute_and_get_children()) == 0):
            ##Calculate Heuristics
            return self.get_heuristic(node)
        if node.state.player == 0:
            maxEval = -math.inf
            for child in node.compute_and_get_children():
                 maxEval = max(maxEval, self.minimax_algorith(child, depth -1, alpha, betta))
                 alpha = max(alpha, maxEval)
                 if alpha >= betta:
                    ##Alpha betta prunning. We stop the interation on the children nodes
                    break
            return maxEval
        else:
            minEval = math.inf
            for child in node.compute_and_get_children():
                minEval = min(minEval, self.minimax_algorith(child, depth -1, alpha, betta))
                betta = min(betta, minEval)
                if betta <= alpha:
                    ##Alpha betta prunning. We stop the interation on the children nodes
                    #Alpha cut-off
                    break
            return minEval

    def get_heuristic(self, node):
        """
        - Distance of hook to closest fish
        - Distance to high value fish
        :return:
        """
        node = node.state

        hook_positions = node.get_hook_positions()
        fish_positions = node.get_fish_positions()
        # fish_scores = node.get_fish_scores()

        # print(f"hook positions: {hook_positions}")
        # print(f"fish positions: {fish_positions}")

        max_pos_x,  max_pos_y = hook_positions[0]

        min_distance = math.inf
        for _, location in fish_positions.items():
            fish_x, fish_y = location
            distance = math.sqrt((fish_x-max_pos_x)**2 + (fish_y-max_pos_y)**2)
            min_distance = min(distance, min_distance)

        return min_distance
