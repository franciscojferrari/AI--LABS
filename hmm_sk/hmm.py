from utils import (
    uniform_random_initialization,
    count_based_initialization,
    baum_welch,
    foward_algorithm_prob,
    forward_algorithm,
    log_PO_given_lambda,
)
from typing import List


class HMM:
    def __init__(
        self,
        nr_states: int,
        nr_emissions: int,
        A: List[List] = None,
        B: List[List] = None,
        pi: List[List] = None,
    ) -> None:
        self.nr_states = nr_states
        self.nr_emissions = nr_emissions
        self.A = A
        self.B = B
        self.pi = pi
        self.log = None
        self.initialize_model()

    def set_matrices(self, A: List[List], B: List[List], pi: List[List]) -> None:
        """Setter method for initial model parameters."""
        self.A = A
        self.B = B
        self.pi = pi

    def initialize_model(self) -> None:
        """Initialize model parameters.

        A and B are initialized using a uniform/random init function, where pi is set with a count
        based init method."""
        if not self.A:
            self.A = uniform_random_initialization(self.nr_states, self.nr_states)
        if not self.B:
            self.B = uniform_random_initialization(self.nr_states, self.nr_emissions)
        if not self.pi:
            self.pi = count_based_initialization(1, self.nr_states, 0.9)

    def train_model(self, O: List, iterations=500) -> None:
        self.A, self.B, self.pi, self.log = baum_welch(
            self.A, self.B, self.pi, O, iterations
        )

    def run_inference(self, O: List, probs: bool = False) -> float:
        """Check if oberservation sequence is likely to be produced by the model
        """
        if probs:
            return foward_algorithm_prob(
                self.A.copy(), self.B.copy(), self.pi.copy(), O.copy(), 60
            )
        else:
            _, scaling_vector = forward_algorithm(
                self.A.copy(), self.B.copy(), self.pi.copy(), O.copy()
            )
            return log_PO_given_lambda(scaling_vector)
