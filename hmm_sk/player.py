#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *

import math
import logging
from typing import List, Tuple
import random as random
from random import randrange
from statistics import mean

LOGGER = logging.getLogger(__name__)


def T(matrix: List[List]) -> List[List]:
    """Transpose of matrix

    Args:
        matrix (list): Matrix dimension n, m

    Returns:
        list: Transponsed matrix of dimension m, n
    """
    return list(map(list, zip(*matrix)))


def elem_wise_product(vector_a, vector_b):
    return [(element_a * element_b) for element_a, element_b in zip(vector_a, vector_b)]


def matrix_mulitplication(a: list, b: list) -> list:
    """Matrix multiplication.

    Arguments:
        a: Matrix with dimension n, m.
        b: Matrix with dimension m, k.

    Returns:
        Matrix with shape n, k.
    """
    return [
        [sum(ii * jj for ii, jj in zip(i, j)) for j in list(map(list, zip(*b)))]
        for i in a
    ]


def parse_input(input_value: str) -> List[List]:
    """Parse input string of HMM0 from kattis

    Args:
        input_value (str): input string

    Returns:
        list: A list with the matrixes [A, B, pi, O]
    """
    matrixes = []
    for idx, rows in enumerate(input_value.splitlines()):
        if len(rows) > 2:
            values = rows.strip().split(" ")
            if idx > 2:
                matrixes.append([int(item) for item in values[1:]])
            else:
                dimensions = int(float(values[1]))
                matrix_values = [float(item) for item in values[2:]]

                matrixes.append(
                    [
                        matrix_values[i: i + dimensions]
                        for i in range(0, len(matrix_values), dimensions)
                    ]
                )
    return matrixes


def foward_algorithm_prob(A: list, B: list, pi: list, O: list, stop_step: int = -1) -> float:
    alpha_list = []
    for _ in range(10):
        num = randrange(len(O) - stop_step)
        new_O = O[num:num + stop_step]
        for index, emission in enumerate(new_O):
            if index == 0:
                alpha = elem_wise_product(pi[0], T(B)[emission])
            else:
                alpha = elem_wise_product(matrix_mulitplication([alpha], A)[0], T(B)[emission])
        alpha_list.append(sum(alpha))

    return mean(alpha_list)


def forward_algorithm(
        A: List[List],
        B: List[List],
        pi: List,
        O: List[List],
        scaled_alpha_matrix: List = [],
        scaling_vector: List = [],
) -> Tuple[List[List], List]:
    """Foward Algo

    Args:
        A (List[List]): Transition Matrix
        B (List[List]): Emission / Output probability matrix
        pi (list): Initial state vector
        alpha (list): vector that represents the probability of being in state j after seeing the first t observations,
        O (list): vector of emissions sequences itself
        first_iteration (bool, optional): states if the function is the first time is running. Defaults to True.

    Returns:
        float: Sum of Alpha
    """

    alpha = list(map(lambda x, y: x * y, pi[0], T(B)[O[0]]))
    c0 = 1 / sum(alpha)

    alpha = list(map(lambda x: x * c0, alpha))

    scaling_vector.append(c0)
    scaled_alpha_matrix.append(alpha)

    for t in range(1, len(O)):
        alpha = []
        ct = 0

        for i in range(len(A)):
            alpha_temp = 0

            for j in range(len(A)):
                alpha_temp += scaled_alpha_matrix[t - 1][j] * A[j][i]
            alpha_temp = alpha_temp * B[i][O[t]]
            alpha.append(alpha_temp)
            ct += alpha_temp
        ct = 1 / ct
        scaling_vector.append(ct)
        alpha = list(map(lambda a: a * ct, alpha))
        scaled_alpha_matrix.append(alpha)

    return scaled_alpha_matrix, scaling_vector


def backward_algorithm(
        A: List[List],
        B: List[List],
        O: List,
        scaling_vector: List,
        scaled_beta_matrix: List[List] = [],
) -> List[List]:
    bt_minus_1 = [scaling_vector[-1] for _ in A]
    scaled_beta_matrix.append(bt_minus_1)

    for t, emission in reversed(list(enumerate(O[:-1]))):
        beta = []

        for i in range(len(A)):
            beta_temp = 0

            for j in range(len(A)):
                beta_temp += A[i][j] * B[j][O[t + 1]] * scaled_beta_matrix[0][j]
            beta_temp = beta_temp * scaling_vector[t]
            beta.append(beta_temp)
        scaled_beta_matrix.insert(0, beta)
    return scaled_beta_matrix


def di_gamma_algorithm(
        A: List[List],
        B: List[List],
        O: List,
        scaled_alpha_matrix: List[List],
        scaled_beta_matrix: List[List],
        gamma_list: List = [],
        di_gamma_list: List = [],
) -> Tuple[List[List], List[List]]:
    for t in range(len(O[:-1])):
        di_gamma = [[] for _ in A]

        for i, a_row in enumerate(A):
            for j in range(len(A)):
                di_gamma_temp = (
                        scaled_alpha_matrix[t][i]
                        * a_row[j]
                        * B[j][O[t + 1]]
                        * scaled_beta_matrix[t + 1][j]
                )
                di_gamma[i].append(di_gamma_temp)
        gamma = [sum(row) for row in di_gamma]
        gamma_list.append(gamma)
        di_gamma_list.append(di_gamma)

    gamma_list.append(scaled_alpha_matrix[-1])

    return gamma_list, di_gamma_list


def re_estimate_pi(gamma_list: List[List]) -> List[List]:
    return [gamma_list[0]]


def re_estimate_A(
        A: List[List], gamma_list: List[List], di_gamma_list: List[List]
) -> List[List]:
    re_estimated_A = []

    for i in range(len(A)):
        re_estimated_A.append([])
        denom = sum([row[i] for row in gamma_list[:-1]])

        for j in range(len(A)):
            number = sum([matrix[i][j] for matrix in di_gamma_list])
            re_estimated_A[i].append(number / denom)

    return re_estimated_A


def re_estimate_B(B: List[List], O: List, gamma_list: List[List]) -> List[List]:
    re_estimated_B = []

    for i in range(len(B)):
        re_estimated_B.append([])
        denom = sum([row[i] for row in gamma_list])
        for j in range(len(B[0])):
            number = sum(
                [vector[i] for t, vector in enumerate(gamma_list) if O[t] == j]
            )
            re_estimated_B[i].append(number / denom)

    return re_estimated_B


def log_PO_given_lambda(scaling_vector: List) -> float:
    return -sum([math.log(ci) for ci in scaling_vector])


def baum_welch(
        A: List[List], B: List[List], pi: List, O: List, maxIters: int
) -> Tuple[List[List], List[List]]:
    iters = 0
    logProb = -99999999999
    oldLogProb = -math.inf

    while iters < maxIters and logProb > oldLogProb and abs(oldLogProb - logProb) > 0.000000005:
        # print(oldLogProb - logProb)
        oldLogProb = logProb

        scaled_alpha_matrix, scaling_vector = forward_algorithm(A, B, pi, O, [], [])
        scaled_beta_matrix = backward_algorithm(A, B, O, scaling_vector, [])

        gamma_list, di_gamma_list = di_gamma_algorithm(
            A, B, O, scaled_alpha_matrix, scaled_beta_matrix, [], []
        )

        pi = re_estimate_pi(gamma_list)
        A = re_estimate_A(A, gamma_list, di_gamma_list)
        B = re_estimate_B(B, O, gamma_list)

        logProb = log_PO_given_lambda(scaling_vector)
        iters += 1

    return A, B, pi, oldLogProb


def baum_welch_exp(
        A: List[List], B: List[List], pi: List, O: List, maxIters: int = 4000
) -> Tuple[List[List], List[List]]:
    iters = 0
    logProb = -99999999999
    oldLogProb = -math.inf
    logprobs = []

    while iters < maxIters and logProb > oldLogProb:
        logprobs.append(logProb)
        oldLogProb = logProb

        scaled_alpha_matrix, scaling_vector = forward_algorithm(A, B, pi, O, [], [])
        scaled_beta_matrix = backward_algorithm(A, B, O, scaling_vector, [])

        gamma_list, di_gamma_list = di_gamma_algorithm(
            A, B, O, scaled_alpha_matrix, scaled_beta_matrix, [], []
        )

        pi = re_estimate_pi(gamma_list)
        A = re_estimate_A(A, gamma_list, di_gamma_list)
        B = re_estimate_B(B, O, gamma_list)

        logProb = log_PO_given_lambda(scaling_vector)
        iters += 1

    return A, B, logprobs[1:], iters, pi


def random_initialization(n: int, m: int):
    matrix = []
    for i in range(n):
        row_temp = [random.random() for _ in range(m)]
        matrix.append([element / sum(row_temp) for element in row_temp])
    return matrix


def diagonal_matrix(n: int):
    print('a')
    matrix = []
    for i in range(n):
        print(i)
        matrix.append([1 if j == i else 0 for j in range(n)])

    return matrix


def uniform_random_initialization(n: int, m: int):
    matrix = []
    for i in range(n):
        row_temp = [random.uniform(9, 10) for _ in range(m)]
        matrix.append([element / sum(row_temp) for element in row_temp])
    return matrix


def uniform_initialization(n: int, m: int):
    return [[1 / m for _ in range(m)] for _ in range(n)]


def count_based_initialization(n: int, m: int, same_state_probability: float = 0.7):
    matrix = []
    for i in range(n):
        matrix.append([])
        for j in range(m):
            if i == j:
                matrix[-1].append(same_state_probability)
            else:
                matrix[-1].append((1 - same_state_probability) / (m - 1))
    return matrix


def pretty_print_matrix(mat: List[List]):
    s = [[str(e) for e in row] for row in mat]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def euclidean_distance(mat_a, mat_b):
    values = [math.pow((a - b), 2) for row_a, row_b in zip(mat_a, mat_b) for a, b in
              zip(row_a, row_b)]
    result = math.sqrt(sum(values) / len(mat_a))

    return result

NR_STATES = 2
INIT_STRATEGY = "default"
TRAIN_ITERATIONS = 35
RETRAINING = False


class DataVault:
    def __init__(self):
        self.observations = None
        self.fish_ids = None
        self.fish_types = [i for i in range(7)]
        self.labels = {}
        self.all_fish_types = []

    def populate_fish_ids(self, nr_fish):
        """Need to know the number of fish in the game"""
        self.fish_ids = [i for i in range(nr_fish)]

    def store_new_observations(self, observations):
        if not self.observations:
            self.observations = [[observation] for observation in observations]

        else:
            self.observations = [
                observation + [new_observation]
                for observation, new_observation in zip(self.observations, observations)
            ]

    def get_fish_observations(self, fish_id):
        return self.observations[int(fish_id)]

    def set_fish_ids(self, fish_ids):
        self.fish_ids = fish_ids

    def get_labels(self):
        return self.labels

    def pop_fish_id(self):
        return self.fish_ids.pop(0)

    def append_fish_type(self, fish_type):
        self.all_fish_types.append(fish_type)

    def get_all_fish_types(self):
        return self.all_fish_types

    def process_guess(self, fish_id, true_type):
        if true_type not in self.labels:
            self.labels[true_type] = fish_id



class ModelVault:
    def __init__(self, nr_of_models_to_train):
        self.nr_of_models_to_train = nr_of_models_to_train
        self.trained_models = []

    def train_and_store_model(self, true_type, data_vault, sequence):

        # find an existing model that can be used for retraining
        if true_type in data_vault.get_all_fish_types() and RETRAINING:
            # print("Training a new model with init from other model")
            idx = data_vault.get_all_fish_types().index(true_type)
            model_for_init = self.trained_models[idx]
            A, B, pi = model_for_init.get_matrices()

            best_model = None
            for _ in range(self.nr_of_models_to_train):
                if best_model == None:
                    model = HMM(NR_STATES, 8)
                    model.train_model(sequence, iterations=TRAIN_ITERATIONS)
                    model.set_matrices(A, B, pi)
                    best_model = model
                else:
                    model = HMM(NR_STATES, 8)
                    model.set_matrices(best_model.A, best_model.B, best_model.pi)
                    model.train_model(sequence, iterations=TRAIN_ITERATIONS)

                    if model.log > best_model.log:
                        best_model = model

        # if there is no existing model yet, use default values
        else:
            # print("Training a new model with random init")
            best_model = None
            for _ in range(self.nr_of_models_to_train):
                if best_model == None:
                    model = HMM(NR_STATES, 8)
                    model.train_model(sequence, iterations=TRAIN_ITERATIONS)
                    best_model = model
                else:
                    model = HMM(NR_STATES, 8)
                    model.set_matrices(best_model.A, best_model.B, best_model.pi)
                    model.train_model(sequence, iterations=TRAIN_ITERATIONS)

                    if model.log > best_model.log:
                        best_model = model
        self.trained_models.append(best_model)

    def predict(self, fish_id,  data_vault):
        sequence = data_vault.get_fish_observations(fish_id)
        try:
            probs = [
                model.run_inference(sequence)
                for model in self.trained_models
            ]
            return probs.index(max(probs))
        except:
            probs = [
                model.run_inference(sequence, prob=True)
                for model in self.trained_models
        ]

            return probs.index(max(probs))


class HMM:
    def __init__(self, nr_states: int, nr_emissions: int) -> None:
        self.nr_states = nr_states
        self.nr_emissions = nr_emissions
        self.A = None
        self.B = None
        self.pi = None
        self.log = None

    def initialize_model(self, init_method: str) -> None:
        """Initialize model parameters"""
        if self.nr_states == 4 and init_method == "halves":
            self.A = [
                [0.25, 0.25, 0, 25, 0.05, 0.05, 0.05, 0.05, 0.05],
                [0.05, 0.05, 0.25, 0.25, 0, 25, 0.05, 0.05, 0.05],
                [0.05, 0.05, 0.05, 0.05, 0.25, 0.25, 0, 25, 0.05],
                [0.25, 0.05, 0.05, 0.05, 0.05, 0.05, 0.25, 0, 25],
            ]
            self.B = uniform_random_initialization(self.nr_states, self.nr_emissions)
            self.pi = uniform_random_initialization(1, self.nr_states)
        if self.nr_states == 4 and init_method == "compass":
            self.A = [
                [0.25, 0.25, 0.05, 0.05, 0.05, 0.05, 0.05, 0.25],
                [0.05, 0.25, 0.25, 0.25, 0.05, 0.05, 0.05, 0.05],
                [0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 0.05, 0.05],
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.25, 0.25, 0, 25],
            ]
            self.B = uniform_random_initialization(self.nr_states, self.nr_emissions)
            self.pi = uniform_random_initialization(1, self.nr_states)
        elif self.nr_states == 8 and not init_method:
            self.A = uniform_random_initialization(self.nr_states, self.nr_emissions)
            self.B = count_based_initialization(self.nr_states, self.nr_emissions)
            self.pi = uniform_random_initialization(1, self.nr_states)
        elif init_method == "default":
            self.A = uniform_random_initialization(self.nr_states, self.nr_states)
            self.B = uniform_random_initialization(self.nr_states, self.nr_emissions)
            self.pi = uniform_random_initialization(1, self.nr_states)

    def train_model(self, O: List, iterations: int=500) -> None:
        self.initialize_model(INIT_STRATEGY)
        self.A, self.B, self.pi, self.log = baum_welch(
            self.A, self.B, self.pi, O, iterations
        )

    def run_inference(self, O: List, prob: bool = False) -> float:
        """Check if oberservation sequence is likely to be produced by the model"""
        if not prob:
            _, scaling_vector = forward_algorithm(
                self.A.copy(), self.B.copy(), self.pi.copy(), O.copy(), [], []
            )
            return log_PO_given_lambda(scaling_vector)
        else:

            return foward_algorithm_prob(
                self.A.copy(), self.B.copy(), self.pi.copy(), O.copy(), 55
            )

    def set_matrices(self, A: List[List], B: List[List], pi: List[List]) -> None:
        self.A = A
        self.B = B
        self.pi = pi

    def get_matrices(self):
        return self.A, self.B, self.pi


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.models_to_train = 20
        self.model_vault = ModelVault(self.models_to_train)
        self.data_vault = DataVault()
        self.labels = None
        self.step = None

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        self.step = step
        self.data_vault.store_new_observations(observations)
        if step == 1:
            self.data_vault.populate_fish_ids(len(observations))
        if step == 108:
            fish_id = self.data_vault.pop_fish_id()
            return fish_id, random.randint(0, 6)
        if step > 108:
            fish_id = self.data_vault.pop_fish_id()
            idx = self.model_vault.predict(fish_id,  self.data_vault)
            pred = self.data_vault.get_all_fish_types()[idx]
            return fish_id, pred

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        self.data_vault.process_guess(fish_id, true_type)
        if self.step >= 108:
            self.model_vault.train_and_store_model(true_type, self.data_vault, self.data_vault.get_fish_observations(fish_id))
            self.data_vault.append_fish_type(true_type)
