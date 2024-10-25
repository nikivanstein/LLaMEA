import numpy as np

class AdaptiveHybridSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, self.budget // dim)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluations = 0

    def differential_evolution(self, func):
        for _ in range(self.budget // self.population_size):
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return
                idxs = np.arange(self.population_size)
                np.random.shuffle(idxs)
                a, b, c = self.population[idxs[:3]]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_vector
                if trial_fitness < func(self.population[i]):
                    self.population[i] = trial_vector

    def local_search(self, func):
        step_size = 0.1
        for _ in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return
                candidate = self.population[i] + step_size * np.random.uniform(-1, 1, self.dim)
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                self.evaluations += 1
                if candidate_fitness < self.best_fitness:
                    self.best_fitness = candidate_fitness
                    self.best_solution = candidate
                if candidate_fitness < func(self.population[i]):
                    self.population[i] = candidate

    def __call__(self, func):
        self.differential_evolution(func)
        self.local_search(func)
        return self.best_solution