import numpy as np

class EfficientAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.f = 0.5
        self.cr = 0.9

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array(list(map(func, population)))
        
        for _ in range(self.budget // self.population_size):
            idxs = np.random.randint(0, self.population_size, (self.population_size, 3))
            a, b, c = population[idxs.T]
            mutants = np.clip(a + self.f * (b - c), -5.0, 5.0)
            crossovers = np.random.rand(self.population_size, self.dim) < self.cr
            trials = np.where(crossovers[:, :, np.newaxis], mutants, population)
            trial_fitness = np.array(list(map(func, trials)))
            
            improved_idx = trial_fitness < fitness
            population[improved_idx] = trials[improved_idx]
            fitness[improved_idx] = trial_fitness[improved_idx]

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        return best_solution, best_fitness