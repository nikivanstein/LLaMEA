import numpy as np

class EnhancedFireflyAlgorithm:
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
            attractiveness_matrix = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * np.linalg.norm(diff_pop, axis=2))
            np.fill_diagonal(attractiveness_matrix, 0)  # Ensuring diagonal elements are 0
            
            steps = self.alpha * (np.random.rand(self.population_size, self.dim) - 0.5)
            
            fitness_comparison = fitness[:, np.newaxis] < fitness  # True where fitness[i] < fitness[j]
            better_indexes = np.where(fitness_comparison)

            movement = np.sum(attractiveness_matrix[:, better_indexes[0]][:, :, np.newaxis] * (population[better_indexes] - population[:, np.newaxis]), axis=1)
            population += movement + steps
            population = np.clip(population, -5.0, 5.0)
            fitness = np.array([func(individual) for individual in population])
        
        best_idx = np.argmin(fitness)
        return population[best_idx]