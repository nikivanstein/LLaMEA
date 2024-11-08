import numpy as np

class EnhancedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = int(np.sqrt(self.budget))
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        evals_per_ind = self.budget // population_size
        best_solution = None
        best_fitness = float('inf')

        for _ in range(population_size):
            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]
            for i in range(len(population)):
                if i != best_idx:
                    population[i] = np.random.uniform(-5.0, 5.0, self.dim)
        
        return best_solution