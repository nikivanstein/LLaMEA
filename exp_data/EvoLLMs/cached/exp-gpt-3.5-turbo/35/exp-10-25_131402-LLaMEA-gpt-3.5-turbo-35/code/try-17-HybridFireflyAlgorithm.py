import numpy as np

class HybridFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 20 * self.dim
        lower_bound = -5.0 * np.ones(self.dim)
        upper_bound = 5.0 * np.ones(self.dim)
        alpha = 0.2
        beta0 = 1.0
        gamma = 0.1
        
        population = np.random.uniform(low=lower_bound, high=upper_bound, size=(pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget - pop_size):
            for i in range(pop_size):
                for j in range(pop_size):
                    if fitness[j] < fitness[i]:
                        attractiveness = beta0 * np.exp(-gamma * np.linalg.norm(population[i] - population[j])**2)
                        step_size = alpha * (np.random.rand(self.dim) - 0.5)
                        population[i] += attractiveness * (population[j] - population[i]) + step_size
                        population[i] = np.clip(population[i], lower_bound, upper_bound)
                        fitness[i] = func(population[i])
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        return best_solution, best_fitness