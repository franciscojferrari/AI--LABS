import math
import random
import time

class MinMaxModel(object):
    def __init__(self, depth):
        self.depth = depth
        self.time_threshold = 75*1e-3

    def best_next_move(self, node):
        startTime = time.time()

        possible_values = {}
        alpha = -math.inf
        for child in node.compute_and_get_children():
            mini_max_result = self.minimax_algorith(child, self.depth, alpha)
            if mini_max_result >= alpha:
                alpha = mini_max_result
                possible_values[child.move] = mini_max_result
        return max(possible_values, key=possible_values.get)

    def minimax_algorith(self, node, depth, alpha = -math.inf, betta = math.inf):

        if depth == 0 or len(node.compute_and_get_children()) == 0:
            # Calculate Heuristics
            return self.get_heuristic(node)
        if node.state.player == 0:
            maxEval = -math.inf
            for child in node.compute_and_get_children():
                 maxEval = max(maxEval, self.minimax_algorith(child, depth -1, alpha, betta))
                 alpha = max(alpha, maxEval)
                 if alpha >= betta:
                    # Alpha betta prunning. We stop the interation on the children nodes
                    break
            return maxEval
        else:
            minEval = math.inf
            for child in node.compute_and_get_children():
                minEval = min(minEval, self.minimax_algorith(child, depth -1, alpha, betta))
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
        node = node.state

        hook_positions = node.get_hook_positions()
        fish_positions = node.get_fish_positions()
        # fish_scores = node.get_fish_scores()

        # print(f"hook positions: {hook_positions}")
        # print(f"fish positions: {fish_positions}")

        # MAX PLAYER SCORE
        max_pos_x,  max_pos_y = hook_positions[0]

        distance_max = math.inf
        for _, location in fish_positions.items():
            fish_x, fish_y = location
            dist = math.sqrt((fish_x-max_pos_x)**2 + (fish_y-max_pos_y)**2)
            distance_max = min(dist, distance_max)

        # print(f"closest fish for max player: {distance_max}")
        # MIN PLAYER SCORE
        max_pos_x, max_pos_y = hook_positions[1]

        distance_min = math.inf
        for _, location in fish_positions.items():
            fish_x, fish_y = location
            dist = math.sqrt((fish_x - max_pos_x) ** 2 + (fish_y - max_pos_y) ** 2)
            distance_min = min(dist, distance_min)

        # print(f"closest fish for min player: {distance_min}")
        heuristic = distance_min - distance_max

        return heuristic

