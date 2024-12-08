import numpy as np

class EnhancedAdaptiveMultistageOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = min(50, self.budget // (2 * dim))
        self.mutation_factor = 0.6
        self.crossover_rate = 0.9
        self.evaluations = 0
        self.best_solution = None
        self.best_fitness = np.inf
        self.dynamic_population()

    def dynamic_population(self):
        self.population_size = int(self.initial_population_size * (1 - self.evaluations / self.budget))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def dynamic_mutation_factor(self):
        return 0.4 + 0.4 * np.random.rand()

    def differential_evolution(self, func):
        while self.evaluations < self.budget and self.population_size > 0:
            new_population = np.empty_like(self.population)
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[idxs]
                mutation_factor = self.dynamic_mutation_factor()
                mutant_vector = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_vector
                new_population[i] = trial_vector if trial_fitness < func(self.population[i]) else self.population[i]
            self.population = new_population
            self.dynamic_population()

    def stochastic_hill_climbing(self, func):
        step_size = 0.05
        while self.evaluations < self.budget and self.population_size > 0:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return
                direction = np.random.uniform(-1, 1, self.dim)
                candidate = self.population[i] + step_size * direction
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                self.evaluations += 1
                if candidate_fitness < self.best_fitness:
                    self.best_fitness = candidate_fitness
                    self.best_solution = candidate
                if candidate_fitness < func(self.population[i]):
                    self.population[i] = candidate
            step_size *= 0.95  # Reduce the step size adaptively
            self.dynamic_population()

    def __call__(self, func):
        self.differential_evolution(func)
        self.stochastic_hill_climbing(func)
        return self.best_solution