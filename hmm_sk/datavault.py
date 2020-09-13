from typing import List


class DataVault:
    def __init__(self) -> None:
        self.observations = None
        self.fish_ids = None
        self.fish_types = [i for i in range(7)]
        self.labels = {}

    def populate_fish_ids(self, nr_fish: int):
        """Need to know the number of fish in the game"""
        self.fish_ids = [i for i in range(nr_fish)]

    def store_new_observations(self, observations: List) -> None:
        if not self.observations:
            self.observations = [[observation] for observation in observations]

        else:
            self.observations = [
                observation + [new_observation]
                for observation, new_observation in zip(self.observations, observations)
            ]

    def get_fish_observations(self, fish_id: int) -> List:
        return self.observations[int(fish_id)]

    def set_fish_ids(self, fish_ids: List) -> None:
        self.fish_ids = fish_ids

    def get_labels(self) -> List:
        return self.labels

    def pop_fish_id(self) -> int:
        return self.fish_ids.pop(0)

    def save_latest_fish_info(self, fish_id: int, true_type: int) -> None:
        if true_type not in self.labels:
            self.labels[true_type] = fish_id
