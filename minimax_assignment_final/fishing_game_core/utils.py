def negamax_algorithm(self, node, depth, alpha=-math.inf, beta=math.inf):
    if time.time() - self.start_time > self.timeout_s:
        self.timeout = True
        return alpha

    if depth == 0:
        return self.get_heuristic(node)
    children = node.compute_and_get_children()
    if len(children) == 0:
        return self.get_heuristic(node)

    value = -math.inf

    for child in children:

        nega_max = -self.negamax_algorithm(child, depth - 1, -alpha, -beta)

        if nega_max > value:
            value = nega_max
        alpha = max(alpha, value)
        if alpha >= beta:
            break

    return value


    # def compute_hash(self, state, reverse=False):
    #     h = 0
    #     hook_positions = state.get_hook_positions()
    #     fish_positions = state.get_fish_positions()
    #
    #     x = 1 if reverse else 0
    #     y = 0 if reverse else 1
    #
    #     for fish_id, location in fish_positions.items():
    #
    #         h ^= self.zobrist_table[location[x]][location[y]][fish_id + 2]
    #
    #     for hook_id, hook_position in hook_positions.items():
    #
    #         h ^= self.zobrist_table[hook_position[x]][hook_position[y]][hook_id]
    #
    #     return h


    # def init_zobrist_table(self):
    #     """Set up zobrist hash table.
    #
    #     Three  dimensional table with the dimensions of unique items x board size x board size"""
    #     self.zobrist_table = [
    #         [
    #             [random.randint(1, 2 ** 64 - 1) for _ in range(self.unique_items)]
    #             for _ in range(self.size_board)
    #         ]
    #         for _ in range(self.size_board)
    #     ]

    # def get_heuristic(self, node):
    #     distance_heuristic = self.get_distance_heuristic(node.state)
    #     score_heuristic = self.get_score_heuristic(node.state)
    #
    #     return distance_heuristic + score_heuristic * 50

    #
    # def get_distance_heuristic(self, state):
    #     hook_positions = state.get_hook_positions()
    #     fish_positions = state.get_fish_positions()
    #     fish_scores = state.fish_scores
    #     scores = []  # heuristics for both players
    #
    #     for player, hook_position in hook_positions.items():
    #         heuristic = 0
    #         for fish_id, location in fish_positions.items():
    #             y_difference = abs(hook_position[1] - location[1])
    #             # get the minimum score of the difference in x
    #             x_difference = min(
    #                 abs(hook_position[0] - location[0]),
    #                 abs(hook_position[0] - location[0] - self.boundary_location),
    #             )
    #
    #             distance = x_difference + y_difference
    #             heuristic += self.fish_values[fish_id] * math.exp(-distance)
    #         scores.append(heuristic)
    #     return scores[0] - scores[1]
    #
    # def get_score_heuristic(self, state):
    #     max_score, min_score = state.get_player_scores()
    #
    #     return max_score - min_score