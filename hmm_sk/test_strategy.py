from utlis import (
    baum_welch,
    uniform_random_inicialization,
    foward_algorithm_prob,
    forward_algorithm,
    log_PO_given_lambda,
    count_based_inicialization,
)
from sklearn.metrics import accuracy_score
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

    def initialize_model(self, init_method, O = None):
        """Initialize model parameters
        TODO: Make more advanced later on
        """

        if self.nr_states == 4 and init_method == "halves":
            print("using halves init")
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
        elif self.nr_states == 8:
            self.A = uniform_random_inicialization(self.nr_states, self.nr_emissions)
            self.B = count_based_inicialization(self.nr_states, self.nr_emissions)
            self.pi = uniform_random_inicialization(1, self.nr_states)
        else:
            self.A = uniform_random_inicialization(self.nr_states, self.nr_states)
            self.B = uniform_random_inicialization(self.nr_states, self.nr_emissions)
            self.pi = uniform_random_inicialization(1, self.nr_states)


    def train_model(self, O, iterations=500):
        self.initialize_model("halves", O)
        self.A, self.B, self.pi = baum_welch(self.A, self.B, self.pi, O, iterations)

    def run_inference(self, O):
        """Check if oberservation sequence is likely to be produced by the model"""
        return foward_algorithm_prob(
            self.A.copy(), self.B.copy(), self.pi.copy(), O.copy(), 15
        )


models = {f"{i}": {"model": None} for i in range(7)}
print(models)

ground_truth, predictions = [], []

for fish_type, sequence in zip(fish_types, sequences):

    if models[str(fish_type)]["model"] == None:
        model = HMM(4, 8)
        model.train_model(sequence[:120], iterations=200)
        models[str(fish_type)]["model"] = model
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
            print(f"probability: {max(probs)}\n")

print(accuracy_score(ground_truth, predictions))
