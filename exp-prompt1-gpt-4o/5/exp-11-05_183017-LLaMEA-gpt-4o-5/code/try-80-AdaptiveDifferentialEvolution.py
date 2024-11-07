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

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.initial_pop_size, self.dim))

    def mutate(self, idx, population):
        candidates = list(range(len(population)))  # Adjust based on dynamic pop size
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        rand_scale = np.random.uniform(0.5, 1.5)
        stochastic_F = self.F * np.random.uniform(0.8, 1.2)  # Add stochastic component
        self.F = np.clip(stochastic_F + self.learning_rate * (self.success_count / max(1, self.total_count) - 0.5), 0.4, 1.0)
        mutant = np.clip(population[a] + self.F * rand_scale * (population[b] - population[c]), self.bounds[0], self.bounds[1])
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, candidate, target, func):
        self.total_count += 1
        if func(candidate) < func(target):
            self.success_count += 1
            return candidate
        return target

    def adapt_parameters(self):
        self.F = np.clip(self.F, 0.5, 1.0)
        if self.total_count > 0:  # Avoid division by zero
            success_rate = self.success_count / self.total_count
            self.CR += self.learning_rate * (success_rate - 0.5)
            self.CR = np.clip(self.CR, 0.8, 1.0)
        if self.evaluations % (self.budget // 4) == 0:  # Periodic reset
            self.CR = 0.9

    def dynamic_population_size(self):
        return max(4, int(self.initial_pop_size * (1 - self.evaluations / self.budget)))

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += self.initial_pop_size

        while self.evaluations < self.budget:
            current_pop_size = self.dynamic_population_size()
            population = population[:current_pop_size]  # Reduce population size dynamically
            fitness = fitness[:current_pop_size]

            for i in range(current_pop_size):
                self.adapt_parameters()
                mutant = self.mutate(i, population)
                trial = self.crossover(population[i], mutant)
                if self.evaluations < self.budget:
                    trial_fitness = func(trial)
                    self.evaluations += 1
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        population[i] = trial

        best_idx = np.argmin(fitness)
        return population[best_idx]