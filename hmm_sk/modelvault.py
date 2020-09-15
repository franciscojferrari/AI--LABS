from hmm import HMM
from datavault import DataVault
from typing import List
import math
from config import NR_STATES, TRAIN_ITERATIONS


class ModelVault:
    def __init__(self, nr_of_models_to_train: int) -> None:
        self.models = {f"{i}": {"model": None} for i in range(7)}
        self.nr_of_models_to_train = nr_of_models_to_train
        self.trained_models = []

    def train_init_models(
        self, data_vault: DataVault, fish_type: int, fish_id: int
    ) -> None:
        if not self.models[str(fish_type)]["model"]:
            self.train_and_store_model(
                fish_type, data_vault.get_fish_observations(fish_id)
            )
            self.trained_models.append(fish_type)

    def train_and_store_model(self, fish_type: int, sequence: List) -> None:
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

    def predict(self, fish_id: int, data_vault: DataVault) -> int:
        sequence = data_vault.get_fish_observations(fish_id)
        probs = [
            model["model"].run_inference(sequence) if model["model"] else -math.inf
            for fish_type, model in self.models.items()
        ]
        return probs.index(max(probs))

    def retrain_models(self, data_vault: DataVault) -> None:
        for model_id in range(len(self.models)):
            fish_id = data_vault.get_labels()[model_id]
            self.train_and_store_model(
                model_id, data_vault.get_fish_observations(fish_id)
            )
