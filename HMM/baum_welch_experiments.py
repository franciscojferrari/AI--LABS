import logging
import sys
from utlis import (
    baum_welch_exp,
    parse_input,
    count_based_initialization,
    random_initialization,
    baum_welch,
    uniform_initialization,
    euclidean_distance,
    uniform_random_initialization,
    foward_algorithm_prob,
    forward_algorithm
)
# from utlis import pretty_print_matrix

import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from statistics import mean


LOGGER = logging.getLogger(__name__)

def pretty_print_matrix(mat):

    s = [[str(e) for e in row] for row in mat]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

def diagonal_matrix(n:int):
    matrix = []
    for i in range(n):
        matrix.append([1 if j == i else 0 for j in range(n)])

    return matrix


def main():
    A, B, pi, O = parse_input(file_content)

    logprob, iterations = [], []
    cos_sim_a_list, cos_sim_b_list = [], []

    for i in range(100, 901, 50):
        LOGGER.info("Emission length: %s.", i)
        logprob.append([])
        iterations.append([])
        cos_sim_a_list.append([])
        cos_sim_b_list.append([])

        for _ in range(10):

            num = randrange(len(O) - i)

            A_new, B_new, logprobs, max_iter, pi_new = baum_welch_exp(
                A, B, pi, O[num : num + i]
            )
            print(f"A_new: {A_new}")
            print(f"A: {A}")
            print(f"pi_new: {pi_new}")
            print(f"pi:  {pi}")

            logprob[-1].append(logprobs)
            iterations[-1].append(max_iter)


    iterations = [mean(iter_list) for iter_list in iterations]
    plt.plot([i for i in range(100, 901, 50)], iterations)
    plt.show()


def q8():
    _A, _B, _pi, O = parse_input(file_content)

    A = count_based_initialization(3, 3, 0.7)
    print(f"A: {A}")
    B = count_based_initialization(3, 4, 0.7)
    print(f"b: {B}")
    pi = uniform_random_initialization(1, 3)

    A_new, B_new, pi_new = baum_welch(A, B, pi, O, 1000)

    print(f"A_new: {np.matrix(A_new)}")
    # print(f"A: {_A}")
    print(f"B_new: {np.matrix(B_new)}")
    # print(f"A: {_B}")
    print(f"pi_new: {np.matrix(pi_new)}")
    # print(f"pi:  {_pi}")

    # print(foward_algorithm_prob(A, B, pi, O))


def q9():
    _A, _B, _pi, O = parse_input(file_content)

    #  experiment with 7 states
    print("\nExperiment  with 7 states:")
    A = uniform_random_initialization(7, 7)
    B = uniform_random_initialization(7, 4)
    pi = uniform_random_initialization(1, 7)

    A_new, B_new, pi_new = baum_welch(A, B, pi, O, 1000)

    print(f"A_new")
    pretty_print_matrix(A_new)
    print(f"B_new")
    pretty_print_matrix(B_new)
    print(f"pi_new")
    pretty_print_matrix(pi_new)

    #  experiment with  6 states
    print("\nExperiment  with 6 states:")
    A = uniform_random_initialization(6, 6)
    B = uniform_random_initialization(6, 4)
    pi = uniform_random_initialization(1, 6)

    A_new, B_new, pi_new = baum_welch(A, B, pi, O, 1000)

    print(f"A_new")
    pretty_print_matrix(A_new)
    print(f"B_new")
    pretty_print_matrix(B_new)
    print(f"pi_new")
    pretty_print_matrix(pi_new)

    #  experiment with  5 states
    print("\nExperiment  with 5 states:")
    A = uniform_random_initialization(5, 5)
    B = uniform_random_initialization(5, 4)
    pi = uniform_random_initialization(1, 5)

    A_new, B_new, pi_new = baum_welch(A, B, pi, O, 1000)

    print(f"A_new")
    pretty_print_matrix(A_new)
    print(f"B_new")
    pretty_print_matrix(B_new)
    print(f"pi_new")
    pretty_print_matrix(pi_new)

    #  experiment with 4 states
    print("\nExperiment  with 4 states:")
    A = uniform_random_initialization(4, 4)
    B = uniform_random_initialization(4, 4)
    pi = uniform_random_initialization(1, 4)

    A_new, B_new, pi_new = baum_welch(A, B, pi, O, 1000)

    print(f"A_new")
    pretty_print_matrix(A_new)
    print(f"B_new")
    pretty_print_matrix(B_new)
    print(f"pi_new")
    pretty_print_matrix(pi_new)

    #  experiment with 3 states
    print("\nExperiment  with 3 states:")
    A = uniform_random_initialization(3, 3)
    B = uniform_random_initialization(3, 4)
    pi = uniform_random_initialization(1, 3)

    A_new, B_new, pi_new = baum_welch(A, B, pi, O, 1000)

    print(f"A_new")
    pretty_print_matrix(A_new)
    print(f"B_new")
    pretty_print_matrix(B_new)
    print(f"pi_new")
    pretty_print_matrix(pi_new)

    #  experiment with 2 states
    print("\nExperiment  with 2 states:")
    A = uniform_random_initialization(2, 2)
    B = uniform_random_initialization(2, 4)
    pi = uniform_random_initialization(1, 2)

    A_new, B_new, pi_new = baum_welch(A, B, pi, O, 1000)

    print(f"A_new")
    pretty_print_matrix(A_new)
    print(f"B_new")
    pretty_print_matrix(B_new)
    print(f"pi_new")
    pretty_print_matrix(pi_new)


def q10():
    _A, _B, _pi, O = parse_input(file_content)
    #  experiment with 3 states
    print("\nExperiment  with 3 states:")
    A = diagonal_matrix(3)
    B = uniform_random_initialization(3, 4)
    pi = [[0, 0, 1]]

    A_new, B_new, pi_new = baum_welch(A, B, pi, O, 1000)

    print(f"A_new")
    pretty_print_matrix(A_new)
    print(f"B_new")
    pretty_print_matrix(B_new)
    print(f"pi_new")
    pretty_print_matrix(pi_new)




if __name__ == "__main__":
    file_content = "".join([text for text in sys.stdin])
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(name)-4s: %(module)-4s :%(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    q10()
