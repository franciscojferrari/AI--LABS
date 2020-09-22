import math
import random
import time


class MinMaxModel(object):
    def __init__(self, depth, init_data):
        self.depth = depth
        self.boundary_location = 20
        self.fish_values = None
        self.time_threshold = 70*1e-3
        self.time_window = 0.7*1e-3
        self.hash_table = {}
        self.init_fish_values(init_data)
        # self.init_zobrist()

    def init_fish_values(self, init_data):
        init_data.pop("game_over")
        updated_fish_values = {}
        indexing = {}
        i = 0
        for fish, value in init_data.items():
            # print(i)
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
       
        
        old_possible_values = {}
        children = node.compute_and_get_children()
        if len(children) == 1:
            return children[0].move

        for increasing_depth in range(self.depth):
            new_possible_values = {}
            if (self.time_threshold - (time.time() - self.startTime) ) <= self.time_window:
                return max(old_possible_values, key=old_possible_values.get)

            alpha = -math.inf
            for child in children:
                if (self.time_threshold - (time.time() - self.startTime) ) <= self.time_window:
                    return max(old_possible_values, key=old_possible_values.get)

                mini_max_result = self.minimax_algorith(child, increasing_depth, alpha)

                if mini_max_result >= alpha:
                    alpha = mini_max_result
                    new_possible_values[child.move] = mini_max_result
            old_possible_values = new_possible_values
        return max(old_possible_values, key=old_possible_values.get)


    def comoute_and_get_children_sorted(self, node, max_player = True):
        children = node.compute_and_get_children()
        children_heuristic = {child:self.get_heuristic(child) for child in children}
        return [key for (key, value) in sorted(children_heuristic.items(), key=lambda x: x[1], reverse = max_player)]

        
    def minimax_algorith(self, node, depth, alpha=-math.inf, betta=math.inf):
        # print(depth)
        if depth == 0 or len(node.compute_and_get_children()) == 0:
            # Calculate Heuristics
            return self.get_heuristic(node)
        
        state_hash = self.computeNewHash(node.state, depth)
        # state_hash = self.computeHash(node.state)
        if state_hash in self.hash_table:
            if self.hash_table[state_hash]['depth'] >= depth:
                return self.hash_table[state_hash]['evaluation']
        
        
        if node.state.player == 0:
            maxEval = -math.inf

            for child in node.compute_and_get_children():
                maxEval = max(
                    maxEval, 
                    self.minimax_algorith(child, depth - 1, alpha, betta)
                )
                alpha = max(alpha, maxEval)

                if alpha >= betta:
                    break # Alpha betta prunning. We stop the interation on the children nodes
                if (self.time_threshold - (time.time() - self.startTime) ) <= self.time_window * self.depth:
                    break

            self.hash_table[state_hash] = {}
            self.hash_table[state_hash]['evaluation'] = maxEval
            self.hash_table[state_hash]['depth'] = depth
            return maxEval
        else:
            minEval = math.inf
            for child in node.compute_and_get_children():
                minEval = min(
                    minEval, self.minimax_algorith(child, depth - 1, alpha, betta)
                )
                betta = min(betta, minEval)
                if betta <= alpha:
                    break  # Alpha cut-off # Alpha betta prunning. We stop the interation on the children nodes
                if (self.time_threshold - (time.time() - self.startTime) ) <= self.time_window * self.depth:
                    break

            self.hash_table[state_hash] = {}
            self.hash_table[state_hash]['evaluation'] = minEval
            self.hash_table[state_hash]['depth'] = depth

            return minEval


    def init_zobrist(self):
        self.zobTable = [[[random.randint(1,2**64 - 1) for i in range(len(self.indexing)+2)]for j in range(self.boundary_location)]for k in range(self.boundary_location )]
    
    def computeHash(self, state):
        h = 0
        hook_positions = state.get_hook_positions()
        fish_positions = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        checking = {}

        for fish_id, position in fish_positions.items():
            x_p, y_p = position

            key_original = str(x_p) + str(y_p)
            key_mirror = str(abs(x_p-20)) + str(y_p)
            checking[key_original] = self.indexing[fish_scores[fish_id]]
            checking[key_mirror] = self.indexing[fish_scores[fish_id]]

        h0_x, h0_y = hook_positions[0]
        checking[str(h0_x)+str(h0_y)] = self.indexing['h0']
        h1_x, h1_y = hook_positions[1]
        checking[str(h1_x)+str(h1_y)] = self.indexing['h1']

        for i in range(self.boundary_location):
            for j in range(self.boundary_location):
                if str(i)+str(j) in checking.keys():
                    h ^= self.zobTable[i][j][checking[str(i)+str(j)]]
        return h

    def computeNewHash(self, state, depth):
        hook_positions = state.get_hook_positions()
        fish_positions = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        checking = {}

        for fish_id, position in fish_positions.items():
            x_p, y_p = position

            key_original = str(x_p) + str(y_p)
            key_mirror = str(abs(x_p-20)) + str(y_p)
            checking[key_original] = self.indexing[fish_scores[fish_id]]
            checking[key_mirror] = self.indexing[fish_scores[fish_id]]

        h0_x, h0_y = hook_positions[0]
        checking[str(h0_x)+str(h0_y)] = self.indexing['h0']
        h1_x, h1_y = hook_positions[1]
        checking[str(h1_x)+str(h1_y)] = self.indexing['h1']
        # checking['depth'] = depth

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
                heuristic += fish_scores[fish_id] * (1 / (distance + 1e-4)) #Having a fish far away is a bad thing so the inverse is usefull
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



