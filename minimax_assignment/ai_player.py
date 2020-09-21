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
        # print(self.indexing)
        self.init_zobrist()
        self.startTime = None

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

    def simple_heuristic(self, state):
        max_score, min_score = state.get_player_scores()
        return max_score - min_score

    def best_next_move(self, node):
        self.startTime = time.time()
        best_move = None

        for depth in range(20):

            possible_values = {}
            alpha = -math.inf
            if time.time() - self.startTime > self.time_threshold:
                # print(f"reaching depth {depth}")
                break
            for child in node.compute_and_get_children():
                if time.time() - self.startTime > self.time_threshold:
                    best_move = max(possible_values, key=possible_values.get)

                mini_max_result = self.minimax_algorith(child, depth, alpha)
                if mini_max_result >= alpha:
                    alpha = mini_max_result
                    possible_values[child.move] = mini_max_result

            best_move = max(possible_values, key=possible_values.get)
        return best_move

    def minimax_algorith(self, node, depth, alpha=-math.inf, betta=math.inf):
        if depth == 0 or len(node.compute_and_get_children()) == 0:
            # Calculate Heuristics
            return self.get_heuristic(node)
        if node.state.player == 0:
            maxEval = -math.inf
            self.get_heuristic(node)
            self.computeHash(node.state)
            for child in node.compute_and_get_children():
                maxEval = max(
                    maxEval, self.minimax_algorith(child, depth - 1, alpha, betta)
                )
                alpha = max(alpha, maxEval)
                if alpha >= betta:
                    # Alpha betta prunning. We stop the interation on the children nodes
                    break
                if time.time() - self.startTime > self.time_threshold:
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
                if time.time() - self.startTime > self.time_threshold:
                    break
            return minEval


    def init_zobrist(self):
        self.zobTable = [[[random.randint(1,2**64 - 1) for i in range(len(self.indexing)+2)]for j in range(self.boundary_location)]for k in range(self.boundary_location )]
    
    def computeHash(self, state):
        h = 0
        hook_positions = state.get_hook_positions()
        fish_positions = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        checking = {}

        for fish_id, possition in fish_positions.items():
            key = ''.join(map(str,possition))
            checking[key] = self.indexing[fish_scores[fish_id]]

        checking[''.join(map(str,hook_positions[0]))] = self.indexing['h0']
        checking[''.join(map(str,hook_positions[1]))] = self.indexing['h1']

        for i in range(self.boundary_location):
            for j in range(self.boundary_location):
                if str(i)+str(j) in checking.keys():
                    h ^= self.zobTable[i][j][checking[str(i)+str(j)]]
        return h
                
                
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



# print(([[[random.randint(1,2**400 - 1) for i in range(12)]for j in range(20)]for k in range(20 )]))