import logging
import sys
from utlis import baum_welch_exp, parse_input
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
        logprob.append([])
        iterations.append([])
        cos_sim_a_list.append([])
        cos_sim_b_list.append([])

        for _ in range(10):

            num = randrange(len(O) - i)

            A_new, B_new, logprobs, max_iter = baum_welch_exp(A, B, pi, O[num:num + i])
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


if __name__ == "__main__":
    file_content = "".join([text for text in sys.stdin])
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(name)-4s: %(module)-4s :%(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
