#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from datavault import DataVault
from modelvault import ModelVault
from constants import *
import random as random
from utlis import *

import math
import logging
from typing import List, Tuple
import random as random
from random import randrange
from statistics import mean

LOGGER = logging.getLogger(__name__)

NR_STATES = 2
TRAIN_ITERATIONS = 30

class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.models = {f"{i}": {"model": None} for i in range(7)}
        self.models_to_train = 20
        self.model_vault = ModelVault(self.models_to_train)
        self.data_vault = DataVault()
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
        if step == 110:
            fish_id = self.data_vault.pop_fish_id()
            return fish_id, random.randint(0, 6)
        if step > 110:
            fish_id = self.data_vault.pop_fish_id()
            pred = self.model_vault.predict(fish_id,  self.data_vault)
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
        if self.step == 110:
            self.model_vault.train_init_models(self.data_vault, true_type, fish_id)
        if self.step > 110:
            if true_type not in self.data_vault.get_labels():
                self.model_vault.train_init_models(self.data_vault, true_type, fish_id)
                self.data_vault.save_latest_fish_info(fish_id, true_type)
            else:
                self.data_vault.save_latest_fish_info(fish_id, true_type)
        if self.step > 120 and len(self.data_vault.get_labels()) > 6:
            if self.step % 10 == 0:
                self.model_vault.retrain_models(self.data_vault)
                


