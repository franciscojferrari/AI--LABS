import logging
import sys
from utlis import baum_welch_exp, parse_input
# import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)

def main():
    A, B, pi, O = parse_input(file_content)

    logprob, iterations = [], []

    for i in  range(100, 901, 100):
        _, _, logprobs, max_iter = baum_welch_exp(A, B, pi, O[:1], 100)
        logprob.append(logprobs)
        iterations.append(max_iter)

    # for values in logprob:
    #     plt.plot(values)
    #
    # plt.show()


if __name__ == "__main__":
    file_content = "".join([text for text in sys.stdin])
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(name)-4s: %(module)-4s :%(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )