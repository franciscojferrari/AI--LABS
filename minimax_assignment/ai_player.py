import math

class MinMaxModel(object):
    def __init__(self, depth):
        self.depth = depth

    def simple_heuristic(self, node):
        # scores = 
        max_score, min_score = node.state.get_player_scores()
        return max_score - min_score

    def best_next_move(self, node):
        possible_values = {}
        for child in node.compute_and_get_children():
            print(child.state.get_hook_positions())
            mini_max_result = self.minimax_algorith(child, self.depth)
            # print(child.state.get_player_scores)
            possible_values[child.move] = mini_max_result
        # print(possible_values)
        return max(possible_values, key=possible_values.get)

    def minimax_algorith(self, node, depth, alpha = -math.inf, betta = math.inf):
        if(depth == 0 or len(node.compute_and_get_children()) == 0):
            ##Calculate Heuristics
            # print(self.simple_heuristic(node))
            # return self.simple_heuristic(node)
            return self.compute_heuristic(node.state)
        # print(depth)
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


    def compute_heuristic(self, state, only_scores=False):
        scores = state.get_player_scores()
        hook_positions = state.get_hook_positions()
        fish_positions = state.get_fish_positions()
        caught_fish = state.get_caught()
        score_based_value = self.get_score_based_value(caught_fish, scores)
        n_fish = len(fish_positions)
        n_caught = int(caught_fish[0] != None) + int(caught_fish[1] != None)
        if n_fish == 0 or n_fish == n_caught:
            if score_based_value > 0:
                return math.inf
            if score_based_value < 0:
                return -math.inf
            return 0.0
        if only_scores:
            return score_based_value
        value_max_player = self.get_proximity_value(hook_positions, fish_positions, caught_fish, self.max_player)
        value_min_player = self.get_proximity_value(hook_positions, fish_positions, caught_fish, 1 - self.max_player)
        proximity_value = value_max_player - value_min_player
        return score_based_value + proximity_value

    def get_score_based_value(self, caught_fish, scores):
        extra_score_max = self.fish_scores[caught_fish[self.max_player]] if caught_fish[self.max_player] is not None else 0
        extra_score_min = self.fish_scores[caught_fish[(1 - self.max_player)]] if caught_fish[(1 - self.max_player)] is not None else 0
        value = 100 * (scores[self.max_player] - scores[(1 - self.max_player)] + extra_score_max - extra_score_min)
        return value

    def get_proximity_value(self, hook_positions, fish_positions, caught_fish, player):
        value = 0.0
        for fish, fish_position in fish_positions.items():
            if fish in caught_fish:
                continue
            else:
                distance_x = min(abs(fish_position[0] - hook_positions[player][0]), self.space_subdivisions - abs(fish_position[0] - hook_positions[player][0]))
                distance_y = abs(fish_position[1] - hook_positions[player][1])
                distance = distance_x + distance_y
            value += float(self.fish_scores[int(fish)]) * math.exp(-2 * distance)

        return value
