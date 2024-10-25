import numpy as np
from scipy.optimize import minimize

class HybridAdaptiveQuantumFirefly:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 14 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.scale_factor_low = 0.4
        self.scale_factor_high = 0.8
        self.evaluations = 0
        self.attraction_coefficient = 1.0
        self.absorption_coefficient = 0.5
        self.dynamic_population_resizing = False

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                trial_vector = self.mutate(i)
                trial_vector = self.firefly_attraction(trial_vector, i)
                if np.random.rand() < 0.3:
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

    def firefly_attraction(self, trial_vector, idx):
        for j in range(self.population_size):
            if self.fitness[j] < self.fitness[idx]:
                distance = np.linalg.norm(self.population[j] - trial_vector)
                beta = self.attraction_coefficient * np.exp(-self.absorption_coefficient * distance ** 2)
                trial_vector += beta * (self.population[j] - trial_vector)
        return np.clip(trial_vector, self.lower_bound, self.upper_bound)

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