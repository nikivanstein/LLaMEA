import numpy as np
from numba import jit, prange

class ParallelizedImprovedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.f = 0.5
        self.cr = 0.9

    @jit(nopython=True, parallel=True)
    def _calculate_fitness(self, func, population):
        fitness = np.zeros(self.population_size)
        for i in prange(self.population_size):
            fitness[i] = func(population[i])
        return fitness

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = self._calculate_fitness(func, population)
        
        for _ in range(self.budget // self.population_size):
            idxs = np.random.randint(self.population_size, size=(self.population_size, 3))
            a, b, c = population[idxs].transpose(1, 0, 2)
            mutants = np.clip(a + self.f * (b - c), -5.0, 5.0)
            crossovers = np.random.rand(self.population_size, self.dim) < self.cr
            trials = np.where(crossovers, mutants, population)
            trial_fitness = self._calculate_fitness(func, trials)
            
            improvements = trial_fitness < fitness
            population = np.where(improvements[:, np.newaxis], trials, population)
            fitness = np.where(improvements, trial_fitness, fitness)

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        return best_solution, best_fitness