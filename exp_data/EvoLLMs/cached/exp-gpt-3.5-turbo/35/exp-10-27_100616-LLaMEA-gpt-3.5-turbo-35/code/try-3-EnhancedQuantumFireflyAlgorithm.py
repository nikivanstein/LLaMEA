import numpy as np

class EnhancedQuantumFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.5  # Quantum probability factor

    def __call__(self, func):
        def attractiveness(i, j):
            return np.exp(-self.alpha * np.linalg.norm(population[i] - population[j]))

        def move_with_fireflies(i):
            for j in range(len(population)):
                if func(population[i]) > func(population[j]) and attractiveness(i, j) > np.random.rand():
                    attraction_factor = 1 / (1 + np.abs(func(population[i]) - func(population[j])))
                    population[i] += attraction_factor * np.random.uniform(-1, 1, self.dim) * np.exp(-self.alpha * np.linalg.norm(population[j] - population[i]))

        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            for i in range(len(population)):
                move_with_fireflies(i)

        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution