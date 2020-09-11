from utlis import (
    baum_welch,
    uniform_random_inicialization,
    foward_algorithm_prob,
    forward_algorithm,
    log_PO_given_lambda,
    count_based_inicialization,
    forward_algorithm,
    log_PO_given_lambda,
)
from sklearn.metrics import accuracy_score
import time
import json

with open("sequences.json") as f:
    data = json.load(f)
    fish_types = data["fish_types"]
    sequences = data["sequences"]

"""
1) Read data
2) Train model on first oberservation and  store model together with 
"""


class HMM:
    def __init__(self, nr_states, nr_emissions):
        self.nr_states = nr_states
        self.nr_emissions = nr_emissions
        self.A = None
        self.B = None
        self.pi = None
        self.log = None

    def initialize_model(self, init_method, O = None):
        """Initialize model parameters
        TODO: Make more advanced later on
        """
        if self.nr_states == 4 and init_method == "halves":
            # print("using halves init")
            self.A = [[0.25, 0.25,  0,25, 0.05, 0.05, 0.05, 0.05, 0.05],
                      [0.05, 0.05, 0.25, 0.25, 0, 25, 0.05, 0.05, 0.05],
                      [0.05, 0.05,  0.05, 0.05, 0.25, 0.25, 0, 25, 0.05],
                      [0.25, 0.05, 0.05, 0.05, 0.05, 0.05, 0.25, 0, 25]]
            self.B = uniform_random_inicialization(self.nr_states, self.nr_emissions)
            self.pi = uniform_random_inicialization(1, self.nr_states)
        if self.nr_states == 4 and init_method == "compass":
            print("using compass init")
            self.A = [[0.25, 0.25, 0.05, 0.05, 0.05, 0.05, 0.05, 0.25],
                      [0.05, 0.25, 0.25, 0.25, 0.05, 0.05, 0.05, 0.05],
                      [0.05, 0.05,  0.05, 0.25, 0.25, 0.25, 0.05, 0.05],
                      [0.05, 0.05, 0.05, 0.05, 0.05, 0.25, 0.25, 0, 25]]
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
        self.A, self.B, self.pi, self.log = baum_welch(self.A, self.B, self.pi, O, iterations)

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




# for nr_states in range(2, 16):
#     accuracies = []
#
#     for _ in range(20):
#
#         models = {f"{i}": {"model": None} for i in range(7)}
#         ground_truth, predictions = [], []
#
#         for fish_type, sequence in zip(fish_types, sequences):
#
#             if models[str(fish_type)]["model"] == None:
#                 model = HMM(nr_states, 8)
#                 model.train_model(sequence[:60], iterations=200)
#                 models[str(fish_type)]["model"] = model
#             else:
#                 probs = [
#                     model["model"].run_inference(sequence)
#                     for fish_type, model in models.items()
#                     if model["model"] != None
#                 ]
#                 if len(probs) == 7:
#                     ground_truth.append(fish_type)
#                     predictions.append(probs.index(max(probs)))
#                     # print(f"fish type: {fish_type}")
#                     # print(f"predicted fish type:  {probs.index(max(probs))}")
#                     # print(f"probability: {max(probs)}\n")
#         accuracies.append(accuracy_score(ground_truth, predictions))
#     print(f"Average accuracy for {nr_states} states: {mean(accuracies)}")


models = {f"{i}": {"model": None} for i in range(7)}
ground_truth, predictions = [], []
nr_of_models_to_train = 20

for fish_type, sequence in zip(fish_types, sequences):


    if models[str(fish_type)]["model"] == None:
        start = time.time()
        best_model = None
        for _ in range(nr_of_models_to_train):
            if best_model  == None:
                print("training first model")
                model = HMM(2, 8)
                model.train_model(sequence[:120], iterations=30)
                best_model = model
            else:
                model = HMM(2, 8)
                model.set_matrices(best_model.A, best_model.B, best_model.pi)
                model.train_model(sequence[:120], iterations=30)

                if model.log > best_model.log:
                    print("new model is better")
                    best_model = model
        models[str(fish_type)]["model"] = best_model
        print(best_model.A)
        print(best_model.B)
        print(best_model.pi)
        print(f"Time took  to  train new model: {time.time() - start}")


    else:
        probs = [
            model["model"].run_inference(sequence)
            for fish_type, model in models.items()
            if model["model"] != None
        ]
        if len(probs) == 7:
            ground_truth.append(fish_type)
            predictions.append(probs.index(max(probs)))
            print(f"fish type: {fish_type}")
            print(f"predicted fish type:  {probs.index(max(probs))}")
            print(probs)
            print(f"probability: {max(probs)}\n")
print(accuracy_score(ground_truth, predictions))
