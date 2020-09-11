#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import math
import logging
from typing import List, Tuple
import random as random
from utlis import baum_welch, uniform_random_inicialization, foward_algorithm_prob, forward_algorithm, log_PO_given_lambda


class PlayerControllerHMM(PlayerControllerHMMAbstract):

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """

        self.sequence_dic = None
        self.models = {i: {"model":None, "fish_type":None} for i in range(7)}
        self.pi = uniform_random_inicialization(1,8)
        pass
    

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        return (step % 70, random.randrange(1,7))

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
        print(correct, fish_id, true_type)
        print("-----------")
        pass