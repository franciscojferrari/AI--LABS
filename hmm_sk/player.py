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
    
    def predict(self):

        O = self.sequence_dic[0]['sequence']
        A = uniform_random_inicialization(4,4)
        B = uniform_random_inicialization(4,8)
        
        new_A, new_B, new_pi = baum_welch(A, B, self.pi, O, 500)
        self.models[0]['model'] = {"A":new_A, "B":new_B, "pi":new_pi}
        self.models[0]['fish_type'] = 6
        # print(foward_algorithm_prob(A, B, pi, O, 15))
        # print(foward_algorithm_prob(new_A, new_B, new_pi, O, 15))
        f, s = forward_algorithm(A.copy(), B.copy(), new_pi.copy(), O.copy(),[],[])
        # print(sum(f[-1])/s[-1])

    def predict1(self, O):

        # O = self.sequence_dic[0]['sequence']

        A = uniform_random_inicialization(4,4)
        B = uniform_random_inicialization(4,8)
        # pi = uniform_random_inicialization(1,8)
        return baum_welch(A, B, self.pi, O, 500)
        # self.models[0]['model'] = {"A":new_A, "B":new_B, "pi":new_pi}
        # self.models[0]['fish_type'] = 6
        # print(foward_algorithm_prob(A, B, pi, O, 15))
        # print(foward_algorithm_prob(new_A, new_B, new_pi, O, 15))

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        if step < 80:
            if not self.sequence_dic:
                # self.sequences = [[i] for i in observations]
                self.sequence_dic = {}
                for i, observation in enumerate(observations):
                    self.sequence_dic[i] = {"sequence": [observation], "fish_type": None, "model":None}
            else:
                for i, observation in enumerate(observations):
                    self.sequence_dic[i]['sequence'].append(observation)

        if step == 80:
            self.predict()  
        if step > 80:
            O = self.sequence_dic[step-81]['sequence']
            new_A, new_B, new_pi = self.predict1(O)
            # print(self.models[0]['model'])
            # state_sequence_prob_ori = foward_algorithm_prob(
            #     self.models[0]['model']['A'].copy(), 
            #     self.models[0]['model']['B'].copy(), 
            #     self.models[0]['model']['pi'].copy(), 
            #     O.copy(), 
            #     10)
            # state_sequence_prob_new =foward_algorithm_prob(
            #     new_A.copy(), 
            #     new_B.copy(), 
            #     new_pi.copy(), 
            #     O.copy(), 
            #     10)
            # # print(state_sequence_prob_ori)
            # print(state_sequence_prob_ori, state_sequence_prob_new)
            alpha_ma, sca_v = forward_algorithm(
                self.models[0]['model']['A'].copy(), 
                self.models[0]['model']['B'].copy(), 
                self.models[0]['model']['pi'].copy(), 
                O.copy(),[], [])
            
            alpha_ma_new, sca_v_new = forward_algorithm(
                new_A.copy(), 
                new_B.copy(), 
                new_pi.copy(), 
                O.copy(),[], [])
            
            # print(sum([j/sca_v[-1] for j in alpha_ma[-1]]), sum([j/sca_v_new[-1] for j in alpha_ma_new[-1]]))
            print(log_PO_given_lambda(sca_v), log_PO_given_lambda(sca_v_new))
            return (step - 81, 6)
        # random_number = random.randint(0, 7 - 1)
        
        # This code would make a random guess on each step:
        if step == 1:
            return (step % 70, 1)

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