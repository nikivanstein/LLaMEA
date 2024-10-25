import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.scale_factor = 0.8
        self.crossover_rate = 0.9
        self.evaluations = 0

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                trial_vector = self.dynamic_mutate(i)
                trial_vector = self.adaptive_crossover(trial_vector, self.population[i])
                trial_vector = self.local_search(trial_vector)
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.population[i] = trial_vector
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.evaluations < self.budget:
                self.fitness[i] = func(self.population[i])
                self.evaluations += 1

    def dynamic_mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        scale_factor_var = self.scale_factor * (1 + np.random.uniform(-0.1, 0.1))
        mutant_vector = self.population[a] + scale_factor_var * (self.population[b] - self.population[c])
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def adaptive_crossover(self, mutant_vector, target_vector):
        crossover_rate_var = self.crossover_rate * (1 + np.random.uniform(-0.05, 0.05))
        crossover = np.random.rand(self.dim) < crossover_rate_var
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        trial_vector = np.where(crossover, mutant_vector, target_vector)
        return trial_vector

    def local_search(self, vector):
        local_vector = vector + np.random.normal(0, 0.1, self.dim)
        return np.clip(local_vector, self.lower_bound, self.upper_bound)