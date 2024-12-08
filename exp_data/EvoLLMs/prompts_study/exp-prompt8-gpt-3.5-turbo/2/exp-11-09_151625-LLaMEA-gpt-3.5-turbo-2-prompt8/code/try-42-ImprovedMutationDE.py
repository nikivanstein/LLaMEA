import numpy as np

class ImprovedMutationDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.CR = 0.9
        self.F = 0.5

    def mutation(self, target, population, diversity, fitness, iteration):
        F_list = np.random.uniform(0, 1, len(fitness))
        F_prob = np.where(fitness == np.min(fitness), 0.9, 0.1)  # Probability based on fitness
        self.F = np.where(F_list < F_prob, np.random.uniform(0, 1), self.F)  # Probabilistic mutation factor
        return self.F