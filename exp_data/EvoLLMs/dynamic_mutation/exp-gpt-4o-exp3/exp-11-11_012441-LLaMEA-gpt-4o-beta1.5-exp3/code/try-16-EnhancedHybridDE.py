import numpy as np

class EnhancedHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.cr = 0.9  # Initial crossover probability
        self.f = 0.8   # Initial differential weight
        self.best_solution = None
        self.best_value = float('inf')
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.population_values = np.full(self.population_size, float('inf'))
        self.function_evals = 0
        self.archive = []

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.population_values[i] == float('inf'):
                self.population_values[i] = func(self.population[i])
                self.function_evals += 1
                if self.population_values[i] < self.best_value:
                    self.best_value = self.population_values[i]
                    self.best_solution = np.copy(self.population[i])

    def mutate(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.f * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.cr
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, target_idx, trial, func):
        trial_value = func(trial)
        self.function_evals += 1
        if trial_value < self.population_values[target_idx]:
            self.archive.append(self.population[target_idx])
            self.population[target_idx] = trial
            self.population_values[target_idx] = trial_value
            if trial_value < self.best_value:
                self.best_value = trial_value
                self.best_solution = np.copy(trial)

    def adapt_params(self, generation):
        oscillation_factor = (1 + np.cos(2 * np.pi * generation / 50)) / 2
        self.cr = 0.5 + 0.4 * oscillation_factor
        self.f = 0.6 + 0.2 * oscillation_factor

    def resize_population(self):
        if self.function_evals > 0.5 * self.budget:
            new_size = max(self.dim, int(self.population_size * 0.7))
            if new_size < self.population_size:
                indices = np.argsort(self.population_values)[:new_size]
                self.population = self.population[indices]
                self.population_values = self.population_values[indices]
                self.population_size = new_size
                self.archive = []

    def __call__(self, func):
        generation = 0
        self.evaluate_population(func)
        while self.function_evals < self.budget:
            self.adapt_params(generation)
            self.resize_population()
            for i in range(self.population_size):
                if self.function_evals >= self.budget:
                    break
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                self.select(i, trial, func)
            generation += 1
        return self.best_solution