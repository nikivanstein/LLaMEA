import numpy as np
from scipy.optimize import minimize

class HybridPSOwithAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocity = np.random.uniform(-1.0, 1.0, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_fitness = np.inf
        self.scale_factor_low = 0.4
        self.scale_factor_high = 0.9
        self.crossover_rate = 0.9
        self.evaluations = 0
        self.local_search_rate = 0.3
    
    def __call__(self, func):
        self.evaluate_population(func)
        self.update_personal_best()
        self.update_global_best()
        
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                self.update_velocity(i)
                self.update_position(i)
                trial_vector = self.mutate(i)
                trial_vector = self.crossover(trial_vector, self.population[i])
                if np.random.rand() < self.local_search_rate:
                    trial_vector = self.adaptive_local_search(trial_vector, func)
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.population[i] = trial_vector
                    self.update_personal_best(i)
            self.update_global_best()
        return self.global_best

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.evaluations < self.budget:
                self.fitness[i] = func(self.population[i])
                self.evaluations += 1

    def update_personal_best(self, i=None):
        if i is None:
            for idx in range(self.population_size):
                if self.fitness[idx] < self.personal_best_fitness[idx]:
                    self.personal_best_fitness[idx] = self.fitness[idx]
                    self.personal_best[idx] = self.population[idx]
        else:
            if self.fitness[i] < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = self.fitness[i]
                self.personal_best[i] = self.population[i]

    def update_global_best(self):
        min_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[min_idx] < self.global_best_fitness:
            self.global_best_fitness = self.personal_best_fitness[min_idx]
            self.global_best = self.personal_best[min_idx]

    def update_velocity(self, i):
        inertia_weight = 0.5
        cognitive_const = 2.0
        social_const = 2.0
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        cognitive_velocity = cognitive_const * r1 * (self.personal_best[i] - self.population[i])
        social_velocity = social_const * r2 * (self.global_best - self.population[i])
        self.velocity[i] = inertia_weight * self.velocity[i] + cognitive_velocity + social_velocity

    def update_position(self, i):
        self.population[i] = np.clip(self.population[i] + self.velocity[i], self.lower_bound, self.upper_bound)

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