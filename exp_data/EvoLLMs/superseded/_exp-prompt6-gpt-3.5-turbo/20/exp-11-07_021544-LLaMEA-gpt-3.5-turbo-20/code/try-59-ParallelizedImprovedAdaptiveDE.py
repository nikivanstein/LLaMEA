import numpy as np
from concurrent.futures import ProcessPoolExecutor

class ParallelizedImprovedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.f = 0.5
        self.cr = 0.9

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        def evaluate_fitness(individual):
            return func(individual)

        fitness = np.array(list(map(evaluate_fitness, population)))
        
        with ProcessPoolExecutor() as executor:
            for _ in range(self.budget // self.population_size):
                idxs = np.random.randint(self.population_size, size=(self.population_size, 3))
                a, b, c = population[idxs].transpose(1, 0, 2)
                mutants = np.clip(a + self.f * (b - c), -5.0, 5.0)
                crossovers = np.random.rand(self.population_size, self.dim) < self.cr
                trials = np.where(crossovers, mutants, population)

                def evaluate_trial_fitness(trial):
                    return func(trial)

                trial_fitness = np.array(list(executor.map(evaluate_trial_fitness, trials)))
                
                improvements = trial_fitness < fitness
                population = np.where(improvements[:, np.newaxis], trials, population)
                fitness = np.where(improvements, trial_fitness, fitness)

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        return best_solution, best_fitness