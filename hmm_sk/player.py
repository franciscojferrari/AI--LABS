#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
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



class DataVault:
    def __init__(self):
        self.observations = None
        self.fish_ids = None
        self.fish_types = [i for i in range(7)]
        self.labels = {}

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

    def save_latest_fish_info(self, fish_id, true_type):
        if true_type not in self.labels:
            self.labels[true_type] = fish_id


class ModelVault:
    def __init__(self, nr_of_models_to_train):
        self.models = {f"{i}": {"model": None} for i in range(7)}
        self.nr_of_models_to_train = nr_of_models_to_train
        self.trained_models = []

    def train_init_models(self, data_vault, fish_type, fish_id):
        if not self.models[str(fish_type)]["model"]:
            self.train_and_store_model(fish_type, data_vault.get_fish_observations(fish_id))
            self.trained_models.append(fish_type)

    def train_and_store_model(self, fish_type, sequence):
        best_model = None
        for _ in range(self.nr_of_models_to_train):
            if best_model == None:
                model = HMM(NR_STATES, 8)
                model.train_model(sequence, iterations=TRAIN_ITERATIONS)
                best_model = model
            else:
                model = HMM(NR_STATES, 8)
                model.train_model(sequence, iterations=TRAIN_ITERATIONS)

                if model.log > best_model.log:
                    best_model = model

        self.models[str(fish_type)]["model"] = best_model

    def predict(self, fish_id, data_vault):
        sequence = data_vault.get_fish_observations(fish_id)
        try:
            probs = [
                model["model"].run_inference(sequence)
                if model['model'] else -math.inf
                for fish_type, model in self.models.items() 
            ]
            return probs.index(max(probs))
        except:
            probs = [
                model["model"].run_inference(sequence, True)
                if model['model'] else 0
                for fish_type, model in self.models.items() 
            ]
            return probs.index(max(probs))

    def retrain_models(self, data_vault):
        for model_id in range(len(self.models)):
            fish_id = data_vault.get_labels()[model_id]
            self.train_and_store_model(model_id, data_vault.get_fish_observations(fish_id))


class HMM:
    def __init__(self, nr_states, nr_emissions, A = None, B = None, pi = None):
        self.nr_states = nr_states
        self.nr_emissions = nr_emissions
        self.A = A
        self.B = B
        self.pi = pi
        self.log = None
        self.initialize_model()

    def initialize_model(self):
        if not self.A:
            self.A = uniform_random_initialization(self.nr_states, self.nr_states)
        if not self.B:
            self.B = uniform_random_initialization(self.nr_states, self.nr_emissions)
        if not self.pi:
            self.pi = count_based_initialization(1, self.nr_states, 0.9)

    def train_model(self, O, iterations=500):
        self.A, self.B, self.pi, self.log = baum_welch(
            self.A, self.B, self.pi, O, iterations
        )

    def run_inference(self, O, probs = False):
        """Check if oberservation sequence is likely to be produced by the model
        """
        if probs:
            return foward_algorithm_prob(
                self.A.copy(), self.B.copy(), self.pi.copy(), O.copy(), 60
            )
        else:
            _, scaling_vector = forward_algorithm(
                self.A.copy(), self.B.copy(), self.pi.copy(), O.copy(), [], []
            )
            return log_PO_given_lambda(scaling_vector)


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
                


