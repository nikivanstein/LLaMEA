import numpy as np

class OptimizedHybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(int(budget / 15), 30)  # Slightly adjusted population size
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(*self.bounds, (self.pop_size, dim))
        self.velocities = np.zeros((self.pop_size, dim))
        self.personal_best_positions = self.population.copy()
        self.personal_best_values = np.full(self.pop_size, np.inf)
        self.global_best_position = np.random.uniform(*self.bounds, dim)
        self.global_best_value = np.inf
        self.evals = 0
        self.F = 0.5  # Updated fixed scaling factor
        self.CR = 0.8  # Reduced crossover rate
        self.omega = 0.4  # Adjusted inertia weight
        self.phi_p = 1.2  # Updated cognitive component
        self.phi_g = 1.6  # Updated social component

    def __call__(self, func):
        self.evaluate_population(func)

        while self.evals < self.budget:
            for i in range(self.pop_size):
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
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

            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            self.velocities = (self.omega * self.velocities +
                               r1 * self.phi_p * (self.personal_best_positions - self.population) +
                               r2 * self.phi_g * (self.global_best_position - self.population))
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