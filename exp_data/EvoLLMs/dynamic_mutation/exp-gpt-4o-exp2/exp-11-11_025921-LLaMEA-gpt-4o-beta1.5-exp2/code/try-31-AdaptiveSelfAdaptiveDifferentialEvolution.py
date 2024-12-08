import numpy as np

class AdaptiveSelfAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.initial_population_size = min(120, 12 * dim)
        self.population_size = self.initial_population_size
        self.F = 0.8
        self.CR = 0.9
        self.convergence_threshold = 0.01
        self.k = 0.5  # New: learning rate for self-adaptation

    def mutation(self, population):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        distance = np.linalg.norm(population[idxs[1]] - population[idxs[2]])
        adaptive_F = self.F * min(1, distance / (self.bounds[1] - self.bounds[0]))
        return population[idxs[0]] + adaptive_F * (population[idxs[1]] - population[idxs[2]])

    def crossover(self, target, mutant, fitness_improvement):
        adaptive_CR = self.CR * (1 + fitness_improvement)
        adaptive_CR = min(1.0, max(0.1, adaptive_CR))
        crossover_mask = np.random.rand(self.dim) < adaptive_CR
        return np.where(crossover_mask, mutant, target)

    def select(self, trial, target, trial_fitness, target_fitness):
        return (trial, trial_fitness) if trial_fitness < target_fitness else (target, target_fitness)

    def adjust_population_size(self, fitness):
        if np.std(fitness) < self.convergence_threshold:
            self.population_size = max(4, self.population_size // 2)
        else:
            self.population_size = min(self.initial_population_size, self.population_size * 2)

    def adapt_parameters(self, fitness, trial_fitness):
        self.F = self.F + self.k * (trial_fitness - np.mean(fitness))
        self.CR = self.CR + self.k * (0.5 - np.random.rand())

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size

        while evaluations < self.budget:
            self.adjust_population_size(fitness)
            for i in range(self.population_size):
                mutant = self.mutation(population)
                mutant = np.clip(mutant, *self.bounds)
                fitness_improvement = (np.min(fitness) - fitness[i]) / (np.max(fitness) - np.min(fitness) + 1e-10)
                trial = self.crossover(population[i], mutant, fitness_improvement)
                trial = np.clip(trial, *self.bounds)

                trial_fitness = func(trial)
                new_candidate, new_fitness = self.select(trial, population[i], trial_fitness, fitness[i])
                evaluations += 1

                if new_fitness < fitness[i]:
                    population[i] = new_candidate
                    fitness[i] = new_fitness
                    self.adapt_parameters(fitness, trial_fitness)

                if evaluations >= self.budget:
                    break

        return population[np.argmin(fitness)]