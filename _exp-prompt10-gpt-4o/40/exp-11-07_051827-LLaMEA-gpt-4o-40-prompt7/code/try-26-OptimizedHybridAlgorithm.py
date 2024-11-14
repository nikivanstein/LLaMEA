import numpy as np

class OptimizedHybridAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(60, budget // 10)  # Adjusted pop_size for better coverage
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(*self.bounds, (self.pop_size, dim))
        self.velocities = np.zeros((self.pop_size, dim))
        self.personal_best_positions = self.population.copy()
        self.personal_best_values = np.inf * np.ones(self.pop_size)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evals = 0
        self.F = 0.6  # Adjusted for better exploration
        self.CR = 0.7  # Adjusted for better recombination
        self.omega = 0.5  
        self.phi_p = 1.5  # Increased to encourage local search
        self.phi_g = 1.3  # Increased for stronger attraction to global best

    def __call__(self, func):
        self.evaluate_population(func)

        while self.evals < self.budget:
            indices = np.random.randint(0, self.pop_size, (self.pop_size, 3))  # Changed to 3 for different mutation
            r1, r2, r3 = indices[:, 0], indices[:, 1], indices[:, 2]
            mutants = np.clip(self.population[r1] + self.F * (self.population[r2] - self.population[r3]), *self.bounds)
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

            r_p, r_g = np.random.rand(2, self.pop_size, self.dim)
            self.velocities = (self.omega * self.velocities +
                               r_p * self.phi_p * (self.personal_best_positions - self.population) +
                               r_g * self.phi_g * (self.global_best_position - self.population))
            self.population = np.clip(self.population + self.velocities, *self.bounds)

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