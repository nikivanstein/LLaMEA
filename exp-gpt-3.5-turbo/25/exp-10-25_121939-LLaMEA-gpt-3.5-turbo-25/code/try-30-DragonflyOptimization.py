import numpy as np

class DragonflyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.step_size = 0.1
        self.attraction_coeff = 0.5
        self.volatility_coeff = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[j] < fitness[i]:
                        distance = np.linalg.norm(population[i] - population[j])
                        attraction = self.attraction_coeff / (1 + distance**2)
                        volatility = np.random.uniform(0, 1, self.dim) * self.volatility_coeff
                        population[i] += attraction * (population[j] - population[i]) + volatility
                        population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
                        fitness[i] = func(population[i])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        return best_solution