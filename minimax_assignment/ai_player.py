import math
import random

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
            # print(child.state.get_hook_positions())
            mini_max_result = self.minimax_algorith(child, self.depth)
            # print(child.state.get_player_scores)
            possible_values[child.move] = mini_max_result
        # print(possible_values)
        if all(value == possible_values[0] for value in possible_values.values()):
            print("random move")
            return random.randint(0, 4)
        print("best move")
        return max(possible_values, key=possible_values.get)

    def minimax_algorith(self, node, depth, alpha = -math.inf, betta = math.inf):
        if(depth == 0 or len(node.compute_and_get_children()) == 0):
            ##Calculate Heuristics
            # print(self.simple_heuristic(node))
            return self.simple_heuristic(node)
            # return self.compute_heuristic(node.state)
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


   
