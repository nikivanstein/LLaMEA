import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.initial_pop_size = 10 * dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.evaluations = 0
        self.learning_rate = 0.05  # Learning rate for adaptive parameters
        self.success_count = 0  # Track successful mutations
        self.total_count = 0  # Track total mutations
        self.stagnation_counter = 0  # Track stagnation

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.initial_pop_size, self.dim))

    def mutate(self, idx, population):
        candidates = list(range(len(population)))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        rand_scale = np.random.uniform(0.5, 1.5)
        self.F = np.clip(self.F + self.learning_rate * (self.success_count / max(1, self.total_count) - 0.5), 0.4, 1.0)
        mutant = np.clip(population[a] + self.F * rand_scale * (population[b] - population[c]), self.bounds[0], self.bounds[1])
        return mutant

    def crossover(self, target, mutant):
        self.adjust_crossover_rate()  # Adjust CR based on stagnation
        crossover_mask = np.random.rand(self.dim) < self.CR
        return np.where(crossover_mask, mutant, target)

    def adjust_crossover_rate(self):
        if self.stagnation_counter > 5:  # If stagnation detected
            self.CR = max(0.7, self.CR - 0.1)  # Decrease CR to encourage exploration
        else:
            self.CR = min(0.9, self.CR + 0.01)  # Slightly increase CR otherwise

    def select(self, candidate, target, func):
        self.total_count += 1
        candidate_fitness = func(candidate)
        target_fitness = func(target)
        self.evaluations += 2
        if candidate_fitness < target_fitness:
            self.success_count += 1
            self.stagnation_counter = 0  # Reset stagnation counter on improvement
            return candidate, candidate_fitness
        else:
            self.stagnation_counter += 1  # Increment stagnation counter
            return target, target_fitness

    def adapt_parameters(self):
        self.F = np.clip(self.F, 0.5, 1.0)
        if self.total_count > 0:
            success_rate = self.success_count / self.total_count
            self.CR += self.learning_rate * (success_rate - 0.5)
            self.CR = np.clip(self.CR, 0.8, 1.0)

    def dynamic_population_size(self):
        return max(4, int(self.initial_pop_size * (1 - self.evaluations / self.budget)))

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += self.initial_pop_size

        while self.evaluations < self.budget:
            current_pop_size = self.dynamic_population_size()
            population = population[:current_pop_size]
            fitness = fitness[:current_pop_size]

            for i in range(current_pop_size):
                self.adapt_parameters()
                mutant = self.mutate(i, population)
                trial = self.crossover(population[i], mutant)
                if self.evaluations < self.budget:
                    population[i], fitness[i] = self.select(trial, population[i], func)

        best_idx = np.argmin(fitness)
        return population[best_idx]