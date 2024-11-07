import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.alpha = 0.2

    def attractiveness(self, x, y):
        return 1.0 / (1.0 + np.linalg.norm(x - y))

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for _ in range(self.budget - len(population)):
            for i in range(len(population)):
                for j in range(len(population)):
                    if fitness[j] < fitness[i]:  # Minimization problem
                        population[i] += self.alpha * (population[j] - population[i]) * self.attractiveness(population[i], population[j])
            
            population = np.clip(population, self.lower_bound, self.upper_bound)
            fitness = np.array([func(x) for x in population])
        
        best_idx = np.argmin(fitness)
        return population[best_idx]