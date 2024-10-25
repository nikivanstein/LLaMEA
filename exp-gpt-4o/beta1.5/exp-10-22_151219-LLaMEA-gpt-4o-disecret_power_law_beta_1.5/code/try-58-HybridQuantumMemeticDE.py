import numpy as np
from scipy.optimize import minimize

class HybridQuantumMemeticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim  # Slightly reduced initial population size
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.scale_factor_low = 0.2  # Slightly lower scale factor range
        self.scale_factor_high = 0.8  # Slightly higher scale factor range
        self.crossover_rate = 0.9  # Slightly higher crossover rate
        self.evaluations = 0
        self.local_search_rate = 0.3  # Slightly increased local search rate
        self.dynamic_population_resizing = True

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            if self.dynamic_population_resizing and self.evaluations > self.budget * 0.3:
                self.update_population_size()
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                trial_vector = self.mutate(i)
                trial_vector = self.crossover(trial_vector, self.population[i])
                if np.random.rand() < self.local_search_rate:
                    trial_vector = self.adaptive_local_search(trial_vector, func)
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

    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_scale_factor = np.random.uniform(self.scale_factor_low, self.scale_factor_high)
        mutant_vector = self.population[a] + adaptive_scale_factor * (self.population[b] - self.population[c])
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def crossover(self, mutant_vector, target_vector):
        crossover = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        trial_vector = np.where(crossover, mutant_vector, target_vector)
        return trial_vector

    def adaptive_local_search(self, vector, func):
        step_size = np.random.normal(0, 0.05, self.dim)
        local_vector = vector + step_size
        local_vector = np.clip(local_vector, self.lower_bound, self.upper_bound)
        if func(local_vector) < func(vector):
            vector = local_vector
            if np.random.rand() < 0.6:
                result = minimize(func, vector, method='BFGS', tol=1e-5)
                if result.success and func(result.x) < func(vector):
                    vector = np.clip(result.x, self.lower_bound, self.upper_bound)
        return vector

    def update_population_size(self):
        reduction_ratio = 0.7  # Slightly different reduction ratio
        if self.evaluations > self.budget * 0.5:
            new_size = max(5 * self.dim, int(self.population_size * reduction_ratio))
            if new_size < self.population_size:
                self.population = self.population[:new_size]
                self.fitness = self.fitness[:new_size]
                self.population_size = new_size