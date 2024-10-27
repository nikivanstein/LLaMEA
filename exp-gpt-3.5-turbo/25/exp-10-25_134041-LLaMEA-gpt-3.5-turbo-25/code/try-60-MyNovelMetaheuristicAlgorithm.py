import numpy as np

class MyNovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pa = 0.25

    def __call__(self, func):
        self.pa = np.clip(self.pa * 1.05, 0, 1)  # Refining the probability of acceptance
        # Implement your novel metaheuristic algorithm here to optimize the black box function 'func'
        return optimized_solution