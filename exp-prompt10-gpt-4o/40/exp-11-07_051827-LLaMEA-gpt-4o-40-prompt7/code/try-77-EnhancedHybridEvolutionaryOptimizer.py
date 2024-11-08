import numpy as np

class EnhancedHybridEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(25, budget // 10)  # Adjusted population size
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(*self.bounds, (self.pop_size, dim))
        self.velocities = np.random.uniform(-0.25, 0.25, (self.pop_size, dim))
        self.personal_best_positions = self.population.copy()
        self.personal_best_values = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evals = 0
        self.F = 0.6  # Adaptive scaling factor
        self.CR = 0.9  # Further increased crossover rate
        self.omega = 0.2  # Reduced inertia weight for faster convergence
        self.phi_p = 1.6  # Enhanced cognitive component
        self.phi_g = 1.4  # Enhanced social component

    def __call__(self, func):
        self.evaluate_population(func)

        while self.evals < self.budget:
            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = indices
                mutant = np.clip(self.population[a] + self.F * (self.population[b] - self.population[c]), *self.bounds)
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                trial_value = func(trial)
                self.evals += 1
                if trial_value < self.personal_best_values[i]:
                    self.personal_best_positions[i] = trial
                    self.personal_best_values[i] = trial_value
                if trial_value < self.global_best_value:
                    self.global_best_value = trial_value
                    self.global_best_position = trial

            rand_vals = np.random.rand(2, self.pop_size, self.dim)
            self.velocities = (self.omega * self.velocities +
                               rand_vals[0] * self.phi_p * (self.personal_best_positions - self.population) +
                               rand_vals[1] * self.phi_g * (self.global_best_position - self.population))
            np.clip(self.population + self.velocities, *self.bounds, out=self.population)

        return self.global_best_value

    def evaluate_population(self, func):
        values = np.apply_along_axis(func, 1, self.population)
        self.evals += self.pop_size
        better_mask = values < self.personal_best_values
        np.copyto(self.personal_best_positions, self.population, where=better_mask[:, np.newaxis])
        np.copyto(self.personal_best_values, values, where=better_mask)
        min_idx = values.argmin()
        if values[min_idx] < self.global_best_value:
            self.global_best_value = values[min_idx]
            self.global_best_position = self.population[min_idx]