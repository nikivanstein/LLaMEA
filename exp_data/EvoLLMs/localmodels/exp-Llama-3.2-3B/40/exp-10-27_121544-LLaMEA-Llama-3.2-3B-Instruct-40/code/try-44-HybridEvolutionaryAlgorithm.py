import numpy as np
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.strategy_prob = 0.4

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        new_individual = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        while True:
            fitness = func(new_individual)
            if np.random.rand() < self.strategy_prob:
                new_individual = self.mutate(new_individual)
            if fitness!= np.nan:
                break

        return new_individual, func(new_individual)

    def mutate(self, individual):
        mutation_rate = 0.1
        new_individual = individual.copy()
        for i in range(self.dim):
            if np.random.rand() < mutation_rate:
                new_individual[i] = np.random.uniform(self.bounds[i][0], self.bounds[i][1])
        return new_individual

# Usage
if __name__ == "__main__":
    func = lambda x: x[0]**2 + x[1]**2
    algorithm = HybridEvolutionaryAlgorithm(100, 2)
    for _ in range(100):
        individual, fitness = algorithm(func)
        print(f"Individual: {individual}, Fitness: {fitness}")