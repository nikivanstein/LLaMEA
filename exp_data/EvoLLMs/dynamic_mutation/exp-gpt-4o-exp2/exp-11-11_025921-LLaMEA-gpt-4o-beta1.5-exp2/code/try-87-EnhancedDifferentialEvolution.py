import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.initial_population_size = min(100, 10 * dim)
        self.population_size = self.initial_population_size
        self.F = 0.8
        self.CR = 0.9
        self.base_convergence_threshold = 0.01
        self.convergence_threshold = self.base_convergence_threshold

    def mutation(self, population, fitness):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        sorted_idx = np.argsort(fitness[idxs])  # Sort by fitness
        best, mid, worst = population[idxs[sorted_idx]]
        diversity_factor = np.std(population, axis=0).mean()
        adaptive_F = self.F * (1 + 0.5 * diversity_factor)  # Adjust F based on diversity
        return best + adaptive_F * (mid - worst)

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
        self.convergence_threshold = max(self.base_convergence_threshold / 2, np.std(fitness) / 10)
        if np.std(fitness) < self.convergence_threshold:
            self.population_size = max(4, self.population_size // 2)
        else:
            self.population_size = min(self.initial_population_size, self.population_size * 2)

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size

        while evaluations < self.budget:
            self.adjust_population_size(fitness)
            for i in range(self.population_size):
                mutant = self.mutation(population, fitness)
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

        return population[np.argmin(fitness)]