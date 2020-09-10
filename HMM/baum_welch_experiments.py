import logging
import sys
from utlis import (
    baum_welch_exp,
    parse_input,
    count_based_inicialization,
    random_inicialization,
    baum_welch,
    uniform_inicialization,
    euclidean_distance,
    uniform_random_inicialization,
    foward_algorithm_prob,
    forward_algorithm,
    diagonal_matrix
)
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from random import randrange
from statistics import mean


LOGGER = logging.getLogger(__name__)


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

            A_new, B_new, logprobs, max_iter = baum_welch_exp(
                A, B, pi, O[num : num + i]
            )
            cos_sim_a = cosine_similarity(A_new, A)
            cos_sim_a = mean([j for i in cos_sim_a for j in i])
            cos_sim_b = cosine_similarity(B_new, B)
            cos_sim_b = mean([j for i in cos_sim_b for j in i])
            logprob[-1].append(logprobs)
            iterations[-1].append(max_iter)
            cos_sim_a_list[-1].append(cos_sim_a)
            cos_sim_b_list[-1].append(cos_sim_b)

    iterations = [mean(iter_list) for iter_list in iterations]
    plt.plot([i for i in range(100, 901, 50)], iterations)
    plt.show()

    cos_sim_a_list = [mean(iter_list) for iter_list in cos_sim_a_list]
    plt.plot([i for i in range(100, 901, 50)], cos_sim_a_list)
    plt.show()

    cos_sim_b_list = [mean(iter_list) for iter_list in cos_sim_b_list]
    plt.plot([i for i in range(100, 901, 50)], cos_sim_b_list)
    plt.show()


def q8():
    _A, _B, _pi, O = parse_input(file_content)

    A = random_inicialization(3, 3)
    B = random_inicialization(3, 4)
    pi = uniform_random_inicialization(1, 3)
    # print(pi)
    # print(euclidean_distance(A, _A))
    A_new, B_new, pi_new = baum_welch(A, B, pi, O, 100)

    print(A_new)
    print(B_new)
    print(pi_new)
    # print(foward_algorithm_prob(_A, _B, _pi, O))
    # foward_algorithm_prob(_A, _B, _pi, O)
    # foward_algorithm_prob(A, B, pi, O)
    # foward_algorithm_prob(A_new, B_new, pi_new, O)
    # print(foward_algorithm_prob(A_new, B_new, pi_new, O))

    # scaled_alpha_matrix, scaling_vector = forward_algorithm(A_new, B_new, pi_new, O, [], [])
    # print(scaled_alpha_matrix[-1])
    # print(scaling_vector[-1])
    # print("----")
    # print(sum(scaled_alpha_matrix[-1])/scaling_vector[-1])

    # print(euclidean_distance(A_new, _A))

def q9():
    _A, _B, _pi, O = parse_input(file_content)
    print(diagonal_matrix)
    A = random_inicialization(5, 5)
    B = random_inicialization(5, 4)
    pi = uniform_random_inicialization(1, 5)
 
    print(A)
    print(B)
    print(pi)

    A_new, B_new, pi_new = baum_welch(A, B, pi, O, 1000)

    print(A_new)
    print(B_new)
    print(pi_new)

    # print(euclidean_distance(A_new, _A))

if __name__ == "__main__":
    file_content = "".join([text for text in sys.stdin])
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(name)-4s: %(module)-4s :%(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    q9()
