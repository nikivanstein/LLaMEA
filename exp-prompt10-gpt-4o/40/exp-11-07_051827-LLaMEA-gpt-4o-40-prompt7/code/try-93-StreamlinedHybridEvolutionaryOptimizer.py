import numpy as np

class StreamlinedHybridEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(25, budget // 15)  # Adjusted population size for better exploration
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(*self.bounds, (self.pop_size, dim))
        self.velocities = np.random.uniform(-0.2, 0.2, (self.pop_size, dim))
        self.personal_best_positions = self.population.copy()
        self.personal_best_values = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evals = 0
        self.F = 0.5  # Slightly reduced scaling factor for stability
        self.CR = 0.85  # Adjusted crossover rate
        self.omega = 0.15  # Lower inertia weight for enhanced exploration
        self.phi_p = 1.8  # Increased cognitive component
        self.phi_g = 1.2  # Reduced social component

    def __call__(self, func):
        self.evaluate_population(func)

        while self.evals < self.budget:
            indices = np.random.choice(self.pop_size, (self.pop_size, 3), replace=True)
            for i, (a, b, c) in enumerate(indices):
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
            cognitive_component = self.phi_p * (self.personal_best_positions - self.population)
            social_component = self.phi_g * (self.global_best_position - self.population)
            self.velocities = self.omega * self.velocities + rand_vals[0] * cognitive_component + rand_vals[1] * social_component
            self.population = np.clip(self.population + self.velocities, *self.bounds)

        return self.global_best_value

    def evaluate_population(self, func):
        values = np.apply_along_axis(func, 1, self.population)
        self.evals += self.pop_size
        better_mask = values < self.personal_best_values
        self.personal_best_positions[better_mask] = self.population[better_mask]
        self.personal_best_values[better_mask] = values[better_mask]
        min_idx = values.argmin()
        if values[min_idx] < self.global_best_value:
            self.global_best_value = values[min_idx]
            self.global_best_position = self.population[min_idx]