import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(*self.bounds, (self.pop_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, dim))
        self.personal_best_positions = self.population.copy()
        self.personal_best_values = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evals = 0
        self.F = 0.5  # Adapted mutation factor
        self.CR = 0.9  # Adapted crossover probability
        self.omega = 0.3 + np.random.rand() / 4  # Slightly adjusted inertia
        self.phi_p = 1.3
        self.phi_g = 1.4

    def __call__(self, func):
        while self.evals < self.budget:
            if self.global_best_position is None or self.evals % (self.pop_size) == 0:
                self.evaluate_population(func)

            for i in range(self.pop_size):
                if self.evals >= self.budget:
                    break

                # Simplified adaptive Differential Evolution
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + self.F * (b - c), *self.bounds)
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                trial_value = func(trial)
                self.evals += 1
                if trial_value < self.personal_best_values[i]:
                    self.personal_best_positions[i], self.personal_best_values[i] = trial, trial_value
                    if trial_value < self.global_best_value:
                        self.global_best_position, self.global_best_value = trial, trial_value

            # Efficient PSO updates
            r_p, r_g = np.random.rand(2, self.pop_size, self.dim)
            self.velocities = (self.omega * self.velocities +
                               self.phi_p * r_p * (self.personal_best_positions - self.population) +
                               self.phi_g * r_g * (self.global_best_position - self.population))
            self.population = np.clip(self.population + self.velocities, *self.bounds)

        return self.global_best_value

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            value = func(self.population[i])
            self.evals += 1
            if value < self.personal_best_values[i]:
                self.personal_best_positions[i], self.personal_best_values[i] = self.population[i], value
                if value < self.global_best_value:
                    self.global_best_value, self.global_best_position = value, self.population[i]