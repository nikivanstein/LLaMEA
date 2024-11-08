import numpy as np

class OptimizedHybridDEAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_values = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evals = 0
        self.F = 0.6
        self.CR = 0.8
        self.omega = 0.4 + np.random.rand() / 5  # Reduced inertia weight range
        self.phi_p = 1.5
        self.phi_g = 1.5

    def __call__(self, func):
        while self.evals < self.budget:
            if self.global_best_position is None or self.evals % (self.pop_size) == 0:
                self.evaluate_population(func)

            for i in range(self.pop_size):
                if self.evals >= self.budget:
                    break

                # Differential Evolution with simplified mutation and crossover
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                mutant = np.clip(self.population[a] + self.F * (self.population[b] - self.population[c]), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])
                
                trial_value = func(trial)
                self.evals += 1
                if trial_value < self.personal_best_values[i]:
                    self.personal_best_positions[i], self.personal_best_values[i] = trial, trial_value
                    if trial_value < self.global_best_value:
                        self.global_best_position, self.global_best_value = trial, trial_value

            # PSO updates with adjusted coefficients
            r_p, r_g = np.random.rand(2, self.pop_size, self.dim)
            self.velocities = (self.omega * self.velocities +
                               self.phi_p * r_p * (self.personal_best_positions - self.population) +
                               self.phi_g * r_g * (self.global_best_position - self.population))
            self.population = np.clip(self.population + self.velocities, self.lower_bound, self.upper_bound)

        return self.global_best_value

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            value = func(self.population[i])
            self.evals += 1
            if value < self.personal_best_values[i]:
                self.personal_best_positions[i], self.personal_best_values[i] = self.population[i], value
                if value < self.global_best_value:
                    self.global_best_value, self.global_best_position = value, self.population[i]