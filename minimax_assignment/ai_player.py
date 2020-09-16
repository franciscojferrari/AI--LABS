import math
import random
import time


class MinMaxModel(object):
    def __init__(self, depth, init_data):
        self.depth = depth
        self.boundary_location = 20
        self.fish_values = None
        self.time_threshold = 65*1e-3
        self.init_fish_values(init_data)
        # print(self.fish_type)
        self.init_zobrist()

    def init_fish_values(self, init_data):
        init_data.pop("game_over")
        updated_fish_values = {}
        fish_type_values = {}
        i = 0
        for fish, value in init_data.items():
            print(i)
            updated_fish_values[int(fish[-1])] = value["score"]
            if value["score"] not in fish_type_values:
                fish_type_values[value["score"]] = i
                i+=1
        self.fish_values = updated_fish_values
        self.fish_type = fish_type_values

    def simple_heuristic(self, state):
        # scores =
        max_score, min_score = state.get_player_scores()
        return max_score - min_score

    def best_next_move(self, node):
        self.computeHash(node.state)
        startTime = time.time()
        possible_values = {}
        alpha = -math.inf
        for child in node.compute_and_get_children():
            if time.time() - startTime > self.time_threshold:
                return max(possible_values, key=possible_values.get)

            mini_max_result = self.minimax_algorith(child, self.depth, alpha)
            if mini_max_result >= alpha:
                alpha = mini_max_result
                possible_values[child.move] = mini_max_result
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
    # def alpha_beta_with_memory(self, node, alpha, beta, depth):
        
    #     if depth == 0:
    #     self.get_heuristic(node)
    #     i
    def init_zobrist(self):
        self.zobTable = [[[random.randint(1,2**4 - 1) for i in range(len(self.fish_type)+2)]for j in range(self.boundary_location)]for k in range(self.boundary_location )]
    def computeHash(self, state):
        h = 0
        hook_positions = state.get_hook_positions()
        fish_positions = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        checking = {}
        for fish_id, possition in fish_positions.items():
            key = ''.join(map(str,possition))
            # print(fish_id)
            # print(self.fish_type)
        #     checking[''.join(possition)] = self.fish_type[fish_id]
        # #     position_xy = fish_positions[fish_id]
        # #     index = self.fish_type[score]
        # print(checking)
        
        # for i in range(self.boundary_location):
        #     for j in range(self.boundary_location):
        #         if element['ij'] == "ij":
        #             h^= self.zobTable[i][j][piece]
                
                
                    



        
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