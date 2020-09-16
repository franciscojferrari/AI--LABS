# prime_mutiprocessing.py

import math
from multiprocessing import Pool
from multiprocessing import freeze_support


import math
import random
import multiprocessing.dummy as mp 


class MinMaxModel(object):
    def __init__(self, depth):
        self.depth = depth

    '''Define function to run mutiple processors and pool the results together'''
    def run_multiprocessing(self, func, i, n_processors):
        with Pool(processes=n_processors) as pool:
            return pool.map(func, i)

    def simple_heuristic(self, node):
        max_score, min_score = node.state.get_player_scores()
        return max_score - min_score

    def best_next_move(self, node):
        freeze_support() 
        possible_values = {}
        out = self.run_multiprocessing(self.minimax_algorith, node.compute_and_get_children(), 4)
        # print(out)
        # alpha = -math.inf

        # for child in node.compute_and_get_children():
        #     mini_max_result = self.minimax_algorith(child, self.depth, alpha)
        #     if mini_max_result >= alpha:
        #         alpha = mini_max_result
        #         possible_values[child.move] = mini_max_result
        return max(possible_values, key=possible_values.get)

    def minimax_algorith(self, node, depth=20, alpha = -math.inf, betta = math.inf):
        if(depth == 0 or len(node.compute_and_get_children()) == 0):
            return self.simple_heuristic(node)

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
