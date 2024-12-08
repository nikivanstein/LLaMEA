import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.initial_population_size = min(100, 10 * dim)
        self.population_size = self.initial_population_size
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.convergence_threshold = 0.01  # New: Threshold for convergence rate

    def mutation(self, population):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        return population[idxs[0]] + self.F * (population[idxs[1]] - population[idxs[2]])

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, trial, target, func):
        return trial if func(trial) < func(target) else target

    def adjust_population_size(self, fitness):
        if np.std(fitness) < self.convergence_threshold:
            self.population_size = max(4, self.population_size // 2)
        else:
            self.population_size = min(self.initial_population_size, self.population_size * 2)

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size

        while evaluations < self.budget:
            self.adjust_population_size(fitness)  # New: Dynamic population size adjustment
            for i in range(self.population_size):
                mutant = self.mutation(population)
                mutant = np.clip(mutant, *self.bounds)
                trial = self.crossover(population[i], mutant)
                trial = np.clip(trial, *self.bounds)

                # Selection
                new_candidate = self.select(trial, population[i], func)
                new_fitness = func(new_candidate)
                evaluations += 1

                if new_fitness < fitness[i]:
                    population[i] = new_candidate
                    fitness[i] = new_fitness

                if evaluations >= self.budget:
                    break

        return population[np.argmin(fitness)]