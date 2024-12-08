import numpy as np

class SelfAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.initial_population_size = min(100, 10 * dim)
        self.population_size = self.initial_population_size
        self.F_base = 0.5  # Base differential weight
        self.CR_base = 0.3  # Base crossover probability
        self.convergence_threshold = 0.01
        self.alpha = 0.5  # Adaptation rate

    def mutation(self, population, F):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        return population[idxs[0]] + F * (population[idxs[1]] - population[idxs[2]])

    def crossover(self, target, mutant, CR):
        crossover_mask = np.random.rand(self.dim) < CR
        return np.where(crossover_mask, mutant, target)

    def select(self, trial, target, func):
        trial_fitness = func(trial)
        target_fitness = func(target)
        return (trial, trial_fitness) if trial_fitness < target_fitness else (target, target_fitness)

    def adjust_parameters(self, fitness, F, CR):
        fitness_improvement = np.std(fitness)
        F = self.F_base + self.alpha * fitness_improvement
        CR = self.CR_base + self.alpha * (1 - fitness_improvement)
        return min(1.0, max(0.1, F)), min(1.0, max(0.1, CR))

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size

        while evaluations < self.budget:
            F, CR = self.adjust_parameters(fitness, self.F_base, self.CR_base)
            self.adjust_population_size(fitness)
            for i in range(self.population_size):
                mutant = self.mutation(population, F)
                mutant = np.clip(mutant, *self.bounds)
                trial = self.crossover(population[i], mutant, CR)
                trial = np.clip(trial, *self.bounds)

                new_candidate, new_fitness = self.select(trial, population[i], func)
                evaluations += 1

                if new_fitness < fitness[i]:
                    population[i] = new_candidate
                    fitness[i] = new_fitness

                if evaluations >= self.budget:
                    break

        return population[np.argmin(fitness)]