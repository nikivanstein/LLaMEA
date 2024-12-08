import numpy as np

class EnhancedParallelFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.1
        self.beta_min = 0.2
        self.gamma = 1.0
        self.attractiveness_matrix = np.zeros((self.population_size, self.population_size))
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            if np.all(self.attractiveness_matrix == 0):
                self.calculate_attractiveness_matrix(population)
            
            steps = self.alpha * (np.random.rand(self.population_size, self.dim) - 0.5)
            
            update = np.zeros_like(population)
            for i in range(self.population_size):
                better_indexes = np.where(fitness < fitness[i])
                update[i] = np.sum(self.attractiveness_matrix[i, better_indexes[0]][:, np.newaxis] * (population[better_indexes] - population[i]), axis=0) + steps[i]
            
            population += update
            population = np.clip(population, -5.0, 5.0)
            fitness = np.array([func(individual) for individual in population])
        
        best_idx = np.argmin(fitness)
        return population[best_idx]
    
    def calculate_attractiveness_matrix(self, population):
        self.attractiveness_matrix = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * np.linalg.norm(population[:, np.newaxis] - population, axis=2))