import numpy as np

class OptimizedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(*self.bounds, (self.pop_size, dim))
        self.velocities = np.zeros((self.pop_size, dim))
        self.personal_best_positions = self.population.copy()
        self.personal_best_values = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evals = 0
        self.F = 0.5  # Adjusted mutated factor
        self.CR = 0.9  # Enhanced crossover probability
        self.omega = 0.5  # Adaptive inertia
        self.phi_p = 1.4
        self.phi_g = 1.6

    def __call__(self, func):
        self.evaluate_population(func)

        adaptive_factor = 1.0
        while self.evals < self.budget:
            indices = np.random.choice(self.pop_size, (self.pop_size, 3), replace=True)
            mutants = np.clip(self.population[indices[:, 0]] + self.F * 
                              (self.population[indices[:, 1]] - self.population[indices[:, 2]]), *self.bounds)
            crossover_mask = np.random.rand(self.pop_size, self.dim) < self.CR
            trials = np.where(crossover_mask, mutants, self.population)

            trial_values = np.apply_along_axis(func, 1, trials)
            self.evals += self.pop_size

            improvements = trial_values < self.personal_best_values
            self.personal_best_positions[improvements] = trials[improvements]
            self.personal_best_values[improvements] = trial_values[improvements]
            if trial_values.min() < self.global_best_value:
                self.global_best_value = trial_values.min()
                self.global_best_position = trials[trial_values.argmin()]

            r_p = np.random.rand(self.pop_size, self.dim)
            r_g = np.random.rand(self.pop_size, self.dim)
            self.velocities = (self.omega * self.velocities +
                               self.phi_p * r_p * (self.personal_best_positions - self.population) +
                               self.phi_g * r_g * (self.global_best_position - self.population))
            adaptive_factor *= 0.95  # Gradually decrease influence
            self.population = np.clip(self.population + adaptive_factor * self.velocities, *self.bounds)

        return self.global_best_value

    def evaluate_population(self, func):
        values = np.apply_along_axis(func, 1, self.population)
        self.evals += self.pop_size
        better_mask = values < self.personal_best_values
        self.personal_best_positions[better_mask] = self.population[better_mask]
        self.personal_best_values[better_mask] = values[better_mask]
        if values.min() < self.global_best_value:
            self.global_best_value = values.min()
            self.global_best_position = self.population[values.argmin()]