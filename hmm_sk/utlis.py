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


def elem_wise_product(vector_a:list, vector_b:list) -> list:
    """[summary]

    Args:
        vector_a (list): vector a
        vector_b (list): vector b

    Returns:
        list: element wise product between vector a and b
    """
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


def foward_algorithm_prob(
    A: List[List],
    B: List[List],
    pi: List,
    O: List[List],
    stop_step: int = 0
) -> float:

    """Foward algorithm without scaling
    We iterate 10 times over different values of alpha calculated over a random set of observations of length stop_step,
     and we return the max of these set of alpha

    Args:
        A (List[List]): Transition Matrix
        B (List[List]): Emission / Output probability matrix
        pi (list): Initial state vector
        O (list): vector of emissions sequences itself
        stop_step (int, optional): number of observations to calculate the state probabily given the output overservations. Defaults to 0

    Returns:
        float: 'belief state': the probability of a state at a certain time, given the history of evidence
    """
    alpha_list = []
    for _ in range(10):
        num = randrange(len(O) - stop_step)
        new_O = O
        if stop_step > 0:
            new_O = O[num:num + stop_step]
        for index, emission in enumerate(new_O):
            if index == 0:
                alpha = elem_wise_product(pi[0], T(B)[emission])
            else:
                alpha = elem_wise_product(matrix_mulitplication([alpha], A)[0], T(B)[emission])
        alpha_list.append(sum(alpha))

    return max(alpha_list)


def forward_algorithm(
        A: List[List],
        B: List[List],
        pi: List,
        O: List[List]
) -> Tuple[List[List], List]:
    """Foward algorithm with scaler.

    Args:
        A (List[List]): Transition Matrix
        B (List[List]): Emission / Output probability matrix
        pi (list): Initial state vector
        O (list): vector of emissions sequences itself
        
    Returns:
        Tuple[List[List], List]: scaled_alpha_matrix (The Matrix with all the alpha matrixes over the different t),
         scaling_vector (The scaling vector for the different alpha matrixes for each time t)
    """
    scaled_alpha_matrix = []
    scaling_vector = []

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
        scaling_vector: List
) -> List[List]:
    """Backwards algorithm with scaler

    Args:
         A (List[List]): Transition Matrix
        B (List[List]): Emission / Output probability matrix
        O (list): vector of emissions sequences itself
        scaling_vector (List): The scaling vector for the different alpha matrixes for each time t.
        scaled_beta_matrix (List[List], optional): The Matrix with all the beta matrixes over the different t.. Defaults to [].

    Returns:
        List[List]: scaled_beta_matrix
    """
    scaled_beta_matrix = []

    bt_minus_1 = [scaling_vector[-1] for _ in A]
    scaled_beta_matrix.append(bt_minus_1)

    for t, _ in reversed(list(enumerate(O[:-1]))):
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
        scaled_beta_matrix: List[List]
) -> Tuple[List[List], List[List]]:
    """Di-Gamma Algortithm

    Args:
        A (List[List]): Transition Matrix
        B (List[List]): Emission / Output probability matrix
        O (list): vector of emissions sequences itself
        scaled_alpha_matrix (List[List]): The Matrix with all the alpha matrixes over the different t
        scaled_beta_matrix (List[List]): The Matrix with all the beta matrixes over the different t

    Returns:
        Tuple[List[List], List[List]]: gamma_list, di_gamma_list
    """
    gamma_list = []
    di_gamma_list = []
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
    """Re-Estimation of PI

    Args:
        gamma_list (List[List]): The list of gamma values

    Returns:
        List[List]: Re-estimated gamma list
    """
    return [gamma_list[0]]


def re_estimate_A(
        A: List[List], gamma_list: List[List], di_gamma_list: List[List]
) -> List[List]:
    """Re-Estimation of A matrix

    Args:
        A (List[List]): A Matrix to be re-estimated
        gamma_list (List[List]): [description]
        di_gamma_list (List[List]): [description]

    Returns:
        List[List]: The new A matrix
    """
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
        oldLogProb = logProb

        scaled_alpha_matrix, scaling_vector = forward_algorithm(A, B, pi, O)
        scaled_beta_matrix = backward_algorithm(A, B, O, scaling_vector)

        gamma_list, di_gamma_list = di_gamma_algorithm(
            A, B, O, scaled_alpha_matrix, scaled_beta_matrix
        )

        pi = re_estimate_pi(gamma_list)
        A = re_estimate_A(A, gamma_list, di_gamma_list)
        B = re_estimate_B(B, O, gamma_list)

        logProb = log_PO_given_lambda(scaling_vector)
        iters += 1

    return A, B, pi, oldLogProb


def random_initialization(n: int, m: int):
    matrix = []
    for _ in range(n):
        row_temp = [random.random() for _ in range(m)]
        matrix.append([element / sum(row_temp) for element in row_temp])
    return matrix


def diagonal_matrix(n: int):
    matrix = []
    for i in range(n):
        print(i)
        matrix.append([1 if j == i else 0 for j in range(n)])

    return matrix


def uniform_random_initialization(n: int, m: int):
    matrix = []
    for _ in range(n):
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
