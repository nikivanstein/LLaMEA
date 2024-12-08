import numpy as np

class FireflyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.2
        self.beta_min = 0.2
        self.gamma = 1.0

    def __call__(self, func):
        pop_size = 10 * self.dim
        lower_bound = -5.0 * np.ones(self.dim)
        upper_bound = 5.0 * np.ones(self.dim)
        
        population = np.random.uniform(low=lower_bound, high=upper_bound, size=(pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget - pop_size):
            for i in range(pop_size):
                for j in range(pop_size):
                    if fitness[j] < fitness[i]:
                        beta = self.beta_min + (1.0 - self.beta_min) * np.exp(-self.gamma * np.linalg.norm(population[j] - population[i])**2)
                        population[i] += beta * (population[j] - population[i]) + self.alpha * np.random.normal(size=self.dim)
                        population[i] = np.clip(population[i], lower_bound, upper_bound)
                        fitness[i] = func(population[i])
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        return best_solution, best_fitness