import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.initial_population_size = min(120, 12 * dim)  # Updated initial population size
        self.population_size = self.initial_population_size
        self.base_F = 0.5  # Base differential weight, modified
        self.CR = 0.85  # Crossover probability, modified
        self.convergence_threshold = 0.01

    def mutation(self, population, generation):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        distance = np.linalg.norm(population[idxs[1]] - population[idxs[2]])
        adaptive_F = self.base_F + 0.3 * np.sin(generation / 10.0)  # New: Adaptive scaling with generation
        adaptive_F *= min(1, distance / (self.bounds[1] - self.bounds[0]))
        return population[idxs[0]] + adaptive_F * (population[idxs[1]] - population[idxs[2]])

    def crossover(self, target, mutant, fitness_improvement):
        adaptive_CR = self.CR * (1 + fitness_improvement)
        adaptive_CR = min(1.0, max(0.1, adaptive_CR))
        crossover_mask = np.random.rand(self.dim) < adaptive_CR
        return np.where(crossover_mask, mutant, target)

    def select(self, trial, target, func):
        trial_fitness = func(trial)
        target_fitness = func(target)
        return (trial, trial_fitness) if trial_fitness < target_fitness else (target, target_fitness)

    def adjust_population_size(self, fitness):
        if np.std(fitness) < self.convergence_threshold:
            self.population_size = max(6, self.population_size // 2)  # Updated minimum size
        else:
            self.population_size = min(self.initial_population_size, int(self.population_size * 1.5))  # Updated growth factor

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size
        generation = 0

        while evaluations < self.budget:
            self.adjust_population_size(fitness)
            for i in range(self.population_size):
                mutant = self.mutation(population, generation)
                mutant = np.clip(mutant, *self.bounds)
                fitness_improvement = (np.min(fitness) - fitness[i]) / (np.max(fitness) - np.min(fitness) + 1e-10)
                trial = self.crossover(population[i], mutant, fitness_improvement)
                trial = np.clip(trial, *self.bounds)

                new_candidate, new_fitness = self.select(trial, population[i], func)
                evaluations += 1

                if new_fitness < fitness[i]:
                    population[i] = new_candidate
                    fitness[i] = new_fitness

                if evaluations >= self.budget:
                    break
            generation += 1

        return population[np.argmin(fitness)]