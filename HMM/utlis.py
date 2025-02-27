import math
import logging
from typing import List, Tuple
import random as random

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
    return [(element_a * element_b) for element_a , element_b in zip(vector_a, vector_b)]

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
                        matrix_values[i : i + dimensions]
                        for i in range(0, len(matrix_values), dimensions)
                    ]
                )
    return matrixes

def foward_algorithm_prob (A:list, B:list, pi:list, O:list) -> float:
    for index, emission in enumerate(O):
        if index == 0:
            alpha = elem_wise_product(pi[0], T(B)[emission])

        else:
            if index == 15:
                print(alpha)
                print(sum(alpha))
            alpha =elem_wise_product( matrix_mulitplication([alpha], A)[0], T(B)[emission])
    
    return sum(alpha)

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

    alpha = list(map(lambda x, y: x*y, pi[0], T(B)[O[0]]))
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
    print(iters)
    return A, B, pi


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

    return A, B, logprobs[1:], iters


def print_list(input_list: List) -> str:
    return_list = [str(i) for i in input_list[::-1]]
    print(" ".join(return_list))


def parse_matrix(matrix: List) -> str:
    rows = len(matrix)
    columns = len(matrix[0])
    list = [rows, columns] + [item for row in matrix for item in row]

    print(" ".join(map(str, list)))

def random_initialization(n:int, m:int):
    matrix = []
    for i in range(n):
        row_temp = [random.random() for _ in range(m)]
        matrix.append([element/sum(row_temp) for element in row_temp])
    return matrix

def uniform_random_initialization(n:int, m:int):
    matrix = []
    for i in range(n):
        row_temp = [random.uniform(9, 10) for _ in range(m)]
        matrix.append([element/sum(row_temp) for element in row_temp])
    return matrix

def diagonal_matrix(n:int):
    matrix = []
    for i in range(n): 
        matrix.append([1 if j==i else 0 for j in range(n)])

def uniform_initialization(n:int, m:int):
    return [[1/m for _ in range(m)] for _ in range(n)]

def count_based_initialization(n:int, m:int, same_state_probability:float = 0.8):
    matrix = []
    for i in range(n):
        matrix.append([])
        for j in range(m):
            if i==j:
                matrix[-1].append(same_state_probability)
            else:
                matrix[-1].append((1-same_state_probability)/(m-1))
    return matrix

def euclidean_distance(mat_a, mat_b):
    col = len(mat_a[0])

    values = [math.pow((a - b), 2) for row_a, row_b in zip(mat_a, mat_b) for a, b in zip(row_a, row_b)]
    result = math.sqrt(sum(values)/len(mat_a))

    return result

