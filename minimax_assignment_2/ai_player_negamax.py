import math
import random
import time


class MinMaxModel(object):
    def __init__(self, depth, init_data):
        self.depth = depth
        self.boundary_location = 20
        self.fish_values = None
        self.time_threshold = 59*1e-3
        
        # self.hash_table_2 = {}
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
        self.fish_values = updated_fish_values
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

                mini_max_value = self.negamax_algorithm(child, increasing_depth, alpha, beta, 1)
                if mini_max_value >= alpha:
                    best_move = child.move
                    alpha = mini_max_value
                if (time.time() - self.startTime) > self.time_threshold:
                    return best_move

        return best_move

    def negamax_algorithm(self, node, depth, alpha= -math.inf, beta= math.inf, color = 1):
        alphaOrig = alpha

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
            value = self.get_heuristic(node) * color
            ttEntry = {'evaluation' : value,  'depth': depth }

            if value <= alpha:
                ttEntry['flag'] = 'LOWERBOUND'
            elif value >= beta:
                ttEntry['flag'] = 'UPPERBOUND'
            else:
                ttEntry['flag'] = 'EXACT'
            self.hash_table[state_hash] = ttEntry   
            return value

        best_value = -math.inf
        random.shuffle(children)
        for child in children:
            best_value = max(best_value, -self.negamax_algorithm(child, depth-1, -alpha, -beta, -color))
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break
            if (time.time() - self.startTime) > self.time_threshold:
                break
        
        ttEntry = {}
        ttEntry['evaluation'] = best_value
        if best_value <= alphaOrig:
            ttEntry['flag'] = 'UPPERBOUND'
        elif best_value >= beta:
            ttEntry['flag'] = 'LOWERBOUND'
        else:
            ttEntry['flag'] = 'EXACT'
        ttEntry['depth'] = depth
        self.hash_table[state_hash] = ttEntry

        # state_hash_2 = self.computeNewHash(node.parent.state)
        # if state_hash_2 in self.hash_table_2:
        #     self.hash_table_2[state_hash_2] = max(self.hash_table_2[state_hash_2], best_value)

        return best_value

    def computeNewHash(self, state):
        hook_positions = state.get_hook_positions()
        fish_positions = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        checking = {}

        for fish_id, position in fish_positions.items():
            x_p, y_p = position

            key_original = str(x_p) + str(y_p)
            checking[key_original] = self.indexing[fish_scores[fish_id]]
            key_mirror = str(abs(x_p-20)) + str(y_p)
            checking[key_mirror] = self.indexing[fish_scores[fish_id]]

        h0_x, h0_y = hook_positions[0]
        checking[str(h0_x)+str(h0_y)] = self.indexing['h0']
        # checking[str(abs(h0_x-20))+str(h0_y)] = self.indexing['h0']

        h1_x, h1_y = hook_positions[1]
        checking[str(h1_x)+str(h1_y)] = self.indexing['h1']
        # checking['player'] = state.player
        # checking[str(abs(h1_x-20))+str(h1_y)] = self.indexing['h1']

        return hash(frozenset(checking.items()))
        
                
    def get_heuristic(self, node):
        """
        - Distance of hook to closest fish
        - Include score of the fish
        - Include score heuristic
        :return:
        """
        distance_heuristic = self.get_distance_heuristic(node.state)
        score_heuristic = self.get_score_heuristic(node.state)
        fish_positions = node.state.get_fish_positions()
        
        n_fish = len(fish_positions)
        fish_caught_max, fish_caught_min = node.state.get_caught()
        n_caught = int(fish_caught_max != None) + int(fish_caught_min != None)

        if n_fish == 0 or n_fish == n_caught:
            if score_heuristic > 0:
                return math.inf
            if score_heuristic < 0:
                return -math.inf
            return 0
        
        return distance_heuristic + score_heuristic * 100 #- player_distance_heuristic*10

    def get_distance_heuristic(self, state):
        hook_positions = state.get_hook_positions()
        fish_positions = state.get_fish_positions()
        fish_scores = state.fish_scores
        scores = []  # heuristics for both players

        for player, hook_position in hook_positions.items():
            heuristic = 0
            for fish_id, location in fish_positions.items():

                y_difference = abs(hook_position[1] - location[1])
                # get the minimum score of the difference in x
                x_difference = min(
                    abs(hook_position[0] - location[0]),
                    abs(hook_position[0] - self.boundary_location - location[0]),
                )

                distance = x_difference + y_difference
                heuristic += fish_scores[fish_id] * math.exp(-2 * distance)
            scores.append(heuristic)
        return scores[0] - scores[1]

    def get_score_heuristic(self, state):
        max_score, min_score = state.get_player_scores()

        fish_caught_max, fish_caught_min = state.get_caught()
        fish_score_max = (self.fish_values[fish_caught_max] if fish_caught_max != None else 0)
        
        fish_score_min = (
            self.fish_values[fish_caught_min] if fish_caught_min != None else 0
        )
        return max_score - min_score + fish_score_max - fish_score_min