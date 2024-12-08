import numpy as np

class QIAEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(low=-5.0, high=5.0, size=(self.budget, self.dim))
        for _ in range(self.budget):
            fitness_values = [func(individual) for individual in population]
            sorted_indices = np.argsort(fitness_values)
            population = population[sorted_indices]

            # Adaptive quantum-inspired operators
            for i in range(self.budget):
                alpha = np.random.uniform(0.9, 1.1)  # Adjust alpha dynamically
                beta = np.random.uniform(0.8, 1.2)  # Adjust beta dynamically
                gamma = np.random.uniform(0.7, 1.3)  # Adjust gamma dynamically
                delta = np.random.uniform(0.6, 1.4)  # Adjust delta dynamically
                population[i] = alpha*population[i] + beta*population[(i+1) % self.budget] + gamma*population[(i+2) % self.budget] + delta*population[(i+3) % self.budget]

        best_solution = population[0]
        return best_solution