import numpy as np

class ImprovedParallelFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.1
        self.beta_min = 0.2
        self.gamma = 1.0

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            diff_pop = population[:, np.newaxis] - population
            norm_diff_pop = np.linalg.norm(diff_pop, axis=2)
            attractiveness_matrix = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * norm_diff_pop)
            steps = self.alpha * (np.random.rand(self.population_size, self.dim) - 0.5)
            
            for i in range(self.population_size):
                better_indexes = np.where(fitness < fitness[i])
                pop_diff = population[better_indexes] - population[i]
                pop_attractiveness = attractiveness_matrix[i, better_indexes[0]][:, np.newaxis]
                population[i] += np.sum(pop_attractiveness * pop_diff, axis=0) + steps[i]
                population[i] = np.clip(population[i], -5.0, 5.0)
                fitness[i] = func(population[i])
        
        best_idx = np.argmin(fitness)
        return population[best_idx]