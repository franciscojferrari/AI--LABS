import math
import random
import time


class MinMaxModel(object):
    def __init__(self, depth, init_data):
        self.depth = depth
        self.boundary_location = 20
        self.fish_values = None
        self.time_threshold = 60*1e-3
        
        self.init_fish_values(init_data)

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
        self.fish_values =  {int(fish.split('fish')[1]):value['score'] for fish, value in init_data.items()}
        indexing['h0'] = i
        indexing['h1'] = i+1
        self.indexing = indexing

    def best_next_move(self, node):
        self.startTime = time.time()
        best_move = 0
        self.hash_table = {}
        
        children = node.compute_and_get_children()
        if len(children) == 1:
            return children[0].move
        alpha = -math.inf
        beta = math.inf
        for increasing_depth in range(self.depth):

            for child in children:
                if (time.time() - self.startTime) > self.time_threshold:
                    return best_move

                mini_max_value = self.minimax_algorith(child, increasing_depth, alpha, beta)
                if mini_max_value >= alpha:
                    best_move = child.move
                    alpha = mini_max_value
        return best_move

    def minimax_algorith(self, node, depth, alpha=-math.inf, beta=math.inf):

        state_hash = self.computeNewHash(node.state)

        if state_hash in self.hash_table:
            ttEntry = self.hash_table[state_hash]
            if ttEntry['depth'] >= depth:

                if ttEntry['flag'] == 'EXACT':
                    return ttEntry['evaluation']
                    
                if ttEntry['flag'] == 'LOWERBOUND' :
                    alpha = max(alpha, ttEntry['evaluation'])
                    
                elif ttEntry['flag'] == 'UPPERBOUND':
                    beta = min(beta, ttEntry['evaluation'])
                
                if alpha >= beta:
                    return ttEntry['evaluation']

        children = node.compute_and_get_children()

        if depth == 0 or len(children) == 0:
            value = self.get_heuristic(node)
            ttEntry = {'evaluation' : value,  'depth': depth }

            if value <= alpha:
                ttEntry['flag'] = 'LOWERBOUND'
            elif value >= beta:
                ttEntry['flag'] = 'UPPERBOUND'
            else:
                ttEntry['flag'] = 'EXACT'
            self.hash_table[state_hash] = ttEntry   
            return value
        
        if node.state.player == 0:
            maxEval = -math.inf
            random.shuffle(children)

            for child in children:
                maxEval = max(
                    maxEval, 
                    self.minimax_algorith(child, depth - 1, alpha, beta)
                )
                alpha = max(alpha, maxEval)

                if alpha >= beta or maxEval == math.inf:
                    break # Alpha beta prunning. We stop the interation on the children nodes
                if (time.time() - self.startTime) > self.time_threshold:
                    break

            ttEntry = {'evaluation' : maxEval,  'depth': depth }
            if maxEval <= alpha:
                ttEntry['flag'] = 'UPPERBOUND'
            elif maxEval >= beta:
                ttEntry['flag'] = 'LOWERBOUND'
            else:
                ttEntry['flag'] = 'EXACT'
            self.hash_table[state_hash] = ttEntry

            return maxEval
        else:
            minEval = math.inf
            random.shuffle(children)
            for child in children:
                minEval = min(
                    minEval, self.minimax_algorith(child, depth - 1, alpha, beta)
                )
                beta = min(beta, minEval)
                if beta <= alpha or minEval == -math.inf:
                    break  # Alpha cut-off # Alpha beta prunning. We stop the interation on the children nodes
                if (time.time() - self.startTime) > self.time_threshold:
                    break
            
            ttEntry = {'evaluation' : minEval,  'depth': depth }
            if minEval >= alpha:
                ttEntry['flag'] = 'UPPERBOUND'
            elif minEval <= beta:
                ttEntry['flag'] = 'LOWERBOUND'
            else:
                ttEntry['flag'] = 'EXACT'
            self.hash_table[state_hash] = ttEntry
            
            return minEval

    def computeNewHash(self, state):
        hook_positions = state.get_hook_positions()
        fish_positions = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        checking = {}

        for fish_id, position in fish_positions.items():
            x_p, y_p = position

            key_original = str(x_p) + str(y_p)
            checking[key_original] = self.indexing[fish_scores[fish_id]]
            key_mirror = str(abs(self.boundary_location - x_p)) + str(y_p)
            checking[key_mirror] = self.indexing[fish_scores[fish_id]]

        h0_x, h0_y = hook_positions[0]
        checking[str(h0_x)+str(h0_y)] = self.indexing['h0']
        # checking[str(abs(self.boundary_location - h0_x))+str(h0_y)] = self.indexing['h0']

        h1_x, h1_y = hook_positions[1]
        checking[str(h1_x)+str(h1_y)] = self.indexing['h1']
        # checking['player'] = state.player
        # checking[str(abs(self.boundary_location - h1_x))+str(h1_y)] = self.indexing['h1']

        return hash(frozenset(checking.items()))
                   
    def get_heuristic(self, node):
        """
        - Distance of hook to closest fish
        - Include score of the fish
        - Include score heuristic
        :return:
        """
        max_score, min_score = node.state.get_player_scores()
        fish_caught_max, fish_caught_min = node.state.get_caught()

        fish_positions = node.state.get_fish_positions()
        hook_positions = node.state.get_hook_positions()

        score_heuristic = self.get_score_heuristic(fish_caught_max, fish_caught_min, max_score, min_score)

        num_fish = len(fish_positions)

        n_caught = int(fish_caught_max != None) + int(fish_caught_min != None)

        if num_fish == 0 or num_fish == n_caught:
            if score_heuristic > 0:
                return math.inf
            if score_heuristic < 0:
                return -math.inf
            return 0.0

        distance_heuristic = self.get_distance_heuristic(hook_positions, fish_positions, fish_caught_max, fish_caught_min)
        return distance_heuristic + score_heuristic * 100

    def get_distance_heuristic(self, hook_positions, fish_positions, fish_caught_max, fish_caught_min):
    
        scores = []  # heuristics for both players
        
        for player, hook_position in hook_positions.items():
            heuristic=0
            for fish_id, fish_location in fish_positions.items():
                if fish_id in (fish_caught_max, fish_caught_min):
                    continue
                else:
                    y_difference = abs(fish_location[1] - hook_position[1])
                    x_difference = min(
                        abs(fish_location[0] - hook_position[0]), 
                        self.boundary_location - abs(fish_location[0] - hook_position[0])
                    )
                    distance = x_difference + y_difference
                heuristic += self.fish_values[fish_id] * math.exp(-2 * distance)
            scores.append(heuristic)
        return scores[0] - scores[1]

    def get_score_heuristic(self, fish_caught_max, fish_caught_min, max_score, min_score):
        
        fish_score_max = (
            self.fish_values[fish_caught_max] if fish_caught_max != None else 0
        )
        
        fish_score_min = (
            self.fish_values[fish_caught_min] if fish_caught_min != None else 0
        )
        return max_score - min_score + fish_score_max - fish_score_min




