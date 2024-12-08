import numpy as np

class MQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(low=-5.0, high=5.0, size=(self.budget, self.dim))
        for _ in range(self.budget):
            fitness_values = [func(individual) for individual in population]
            sorted_indices = np.argsort(fitness_values)
            population = population[sorted_indices]

            # Apply modified quantum-inspired operators
            for i in range(self.budget):
                alpha = np.random.uniform()
                beta = np.random.uniform()
                gamma = np.random.uniform()
                delta = np.random.uniform()
                epsilon = np.random.uniform()  # New operator introduced
                population[i] = alpha*population[i] + beta*population[(i+1) % self.budget] + gamma*population[(i+2) % self.budget] + delta*population[(i+3) % self.budget] + epsilon*np.random.normal(size=self.dim)

        best_solution = population[0]
        return best_solution