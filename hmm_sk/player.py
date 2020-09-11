#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import math
import logging
from typing import List, Tuple
import random as random
from utlis import (
    baum_welch,
    uniform_random_inicialization,
    foward_algorithm_prob,
    forward_algorithm,
    log_PO_given_lambda,
    count_based_inicialization,
)


class DataVault:
    def __init__(self):
        self.observations = None

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


class ModelVault:
    def __init__(self, nr_of_models_to_train):
        self.models = {f"{i}": {"model": None} for i in range(7)}
        self.nr_of_models_to_train = nr_of_models_to_train
        self.labels = None
        self.trained_models = []

    def train_init_models(self, data_vault):
        for fish_type, model in self.models.items():
            if not model["model"]:
                self.train_and_store_model(fish_type, data_vault.get_fish_observations(self.labels[int(fish_type)]))

    def train_and_store_model(self, fish_type, sequence):
        best_model = None
        for _ in range(self.nr_of_models_to_train):
            if best_model == None:
                print("training first model")
                model = HMM(2, 8)
                model.train_model(sequence[:90], iterations=30)
                best_model = model
            else:
                model = HMM(2, 8)
                model.set_matrices(best_model.A, best_model.B, best_model.pi)
                model.train_model(sequence[:90], iterations=30)

                if model.log > best_model.log:
                    best_model = model
        print("model trained")
        self.models[str(fish_type)]["model"] = best_model

    def set_labels(self, labels):
        self.labels = labels

    def print_models(self):
        print(self.models)


class GuessingMachine:
    def __init__(self):
        self.fish_types = [i for i in range(7)]
        self.fish_ids = None
        self.labels = {i: None for i in range(7)}
        self.correct_fish = 0

    def populate_fish_ids(self, nr_fish):
        """Need to know the number of fish in the game"""
        self.fish_ids = [i for i in range(nr_fish)]

    def what_to_guess(self):
        fish_type = random.choice(self.fish_types)
        fish_id = random.choice(self.fish_ids)
        return fish_id, fish_type

    def process_guess(self, _, fish_id, true_type):
        if true_type in self.fish_types:
            self.fish_types.remove(true_type)
            self.fish_ids.remove(fish_id)
            self.labels[true_type] = fish_id
            self.correct_fish += 1

    def get_labels(self):
        return self.labels

    def get_correct_fish(self):
        return self.correct_fish


class HMM:
    def __init__(self, nr_states, nr_emissions):
        self.nr_states = nr_states
        self.nr_emissions = nr_emissions
        self.A = None
        self.B = None
        self.pi = None
        self.log = None

    def initialize_model(self, init_method, O=None):
        """Initialize model parameters
        TODO: Make more advanced later on
        """
        if self.nr_states == 4 and init_method == "halves":
            # print("using halves init")
            self.A = [
                [0.25, 0.25, 0, 25, 0.05, 0.05, 0.05, 0.05, 0.05],
                [0.05, 0.05, 0.25, 0.25, 0, 25, 0.05, 0.05, 0.05],
                [0.05, 0.05, 0.05, 0.05, 0.25, 0.25, 0, 25, 0.05],
                [0.25, 0.05, 0.05, 0.05, 0.05, 0.05, 0.25, 0, 25],
            ]
            self.B = uniform_random_inicialization(self.nr_states, self.nr_emissions)
            self.pi = uniform_random_inicialization(1, self.nr_states)
        if self.nr_states == 4 and init_method == "compass":
            # print("using compass init")
            self.A = [
                [0.25, 0.25, 0.05, 0.05, 0.05, 0.05, 0.05, 0.25],
                [0.05, 0.25, 0.25, 0.25, 0.05, 0.05, 0.05, 0.05],
                [0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 0.05, 0.05],
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.25, 0.25, 0, 25],
            ]
            self.B = uniform_random_inicialization(self.nr_states, self.nr_emissions)
            self.pi = uniform_random_inicialization(1, self.nr_states)
        elif self.nr_states == 8 and not init_method:
            self.A = uniform_random_inicialization(self.nr_states, self.nr_emissions)
            self.B = count_based_inicialization(self.nr_states, self.nr_emissions)
            self.pi = uniform_random_inicialization(1, self.nr_states)
        elif init_method == "default":
            self.A = uniform_random_inicialization(self.nr_states, self.nr_states)
            self.B = uniform_random_inicialization(self.nr_states, self.nr_emissions)
            self.pi = uniform_random_inicialization(1, self.nr_states)

    def train_model(self, O, iterations=500):
        self.initialize_model("default", O)
        self.A, self.B, self.pi, self.log = baum_welch(
            self.A, self.B, self.pi, O, iterations
        )

    def run_inference(self, O):
        """Check if oberservation sequence is likely to be produced by the model
        TODO:  Change  to  forward_algorithm"""
        # return foward_algorithm_prob(
        #     self.A.copy(), self.B.copy(), self.pi.copy(), O.copy(), 50
        # )

        _, scaling_vector = forward_algorithm(
            self.A.copy(), self.B.copy(), self.pi.copy(), O.copy(), [], []
        )
        return log_PO_given_lambda(scaling_vector)

    def set_matrices(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.sequence_dic = None
        self.models = {f"{i}": {"model": None} for i in range(7)}
        self.models_to_train = 20
        self.start_guessing_step = 90
        self.guess_machine = GuessingMachine()
        self.model_vault = ModelVault(self.models_to_train)
        self.data_vault = DataVault()
        self.labels = None

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        self.data_vault.store_new_observations(observations)
        # print(step)
        if step == 1:
            self.guess_machine.populate_fish_ids(len(observations))
            fish_id, fish_type = self.guess_machine.what_to_guess()
            return fish_id, fish_type
        elif 1 < step < 40:
            if self.guess_machine.get_correct_fish() < 7:
                fish_id, fish_type = self.guess_machine.what_to_guess()
                print("making  guess")
                return fish_id, fish_type
            else:
                if not self.labels:
                    self.labels = self.guess_machine.get_labels()
                    self.model_vault.set_labels(self.labels)

        elif 90 < step < 100:
            self.model_vault.train_init_models(self.data_vault)
        else:
            pass

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
        self.guess_machine.process_guess(correct, fish_id, true_type)
