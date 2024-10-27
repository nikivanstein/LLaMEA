import numpy as np

class MothFlameOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.2
        self.beta0 = 1.0
        self.gamma = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[j] < fitness[i]:  # Minimization
                        r = np.linalg.norm(population[i] - population[j])
                        beta = self.beta0 * np.exp(-self.gamma * r**2)
                        population[i] += beta * (population[j] - population[i]) + self.alpha * np.random.uniform(-1, 1, self.dim)
                        population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
                        fitness[i] = func(population[i])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        return best_solution