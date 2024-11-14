import numpy as np

class MultiObjectiveNSGAII:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.max_generations = budget // self.population_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def non_dominated_sort(self, population):
        # Non-dominated sorting implementation
        ...

    def crowding_distance(self, population):
        # Crowding distance calculation implementation
        ...

    def binary_tournament_selection(self, population):
        # Binary tournament selection implementation
        ...

    def __call__(self, func):
        # NSGA-II optimization implementation
        ...