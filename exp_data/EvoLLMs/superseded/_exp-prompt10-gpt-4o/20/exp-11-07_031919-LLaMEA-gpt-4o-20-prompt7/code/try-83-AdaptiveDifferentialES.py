import numpy as np

class AdaptiveDifferentialES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.best = None
        self.best_fitness = np.inf
        self.beta_max = 0.9  # Modified for better exploration
        self.beta_min = 0.3  # Modified for better exploration
        self.alpha = 0.8  # Modified to allow more crossover
        self.current_budget = 0

    def __call__(self, func):
        self.evaluate_population(func)
        while self.current_budget < self.budget:
            self.perform_differential_evolution(func)
            self.apply_adaptive_mutation(func)
        return self.best

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.current_budget >= self.budget:
                break
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.current_budget += 1
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best = self.population[i].copy()

    def perform_differential_evolution(self, func):
        for i in range(self.population_size):
            if self.current_budget >= self.budget:
                break
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = indices
            beta = self.beta_min + (self.beta_max - self.beta_min) * (1 - self.current_budget / self.budget)
            mutant = np.clip(self.population[a] + beta * (self.population[b] - self.population[c]), self.lower_bound, self.upper_bound)
            trial = np.where(np.random.rand(self.dim) < self.alpha, mutant, self.population[i])
            trial_fitness = func(trial)
            self.current_budget += 1
            if trial_fitness < self.fitness[i]:
                self.population[i] = trial
                self.fitness[i] = trial_fitness
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best = trial

    def apply_adaptive_mutation(self, func):
        adapt_scale = (2 * (self.budget - self.current_budget) / self.budget)  # Enhanced scaling for mutation
        for i in range(self.population_size):
            if self.current_budget >= self.budget:
                break
            candidate = self.population[i] + np.random.normal(0, 0.05 * adapt_scale, self.dim)
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)
            self.current_budget += 1
            if candidate_fitness < self.fitness[i]:
                self.population[i] = candidate
                self.fitness[i] = candidate_fitness
                if candidate_fitness < self.best_fitness:
                    self.best_fitness = candidate_fitness
                    self.best = candidate