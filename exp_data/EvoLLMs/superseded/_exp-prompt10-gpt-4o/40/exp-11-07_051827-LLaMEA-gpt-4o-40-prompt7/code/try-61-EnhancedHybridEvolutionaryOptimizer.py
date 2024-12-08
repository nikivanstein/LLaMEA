import numpy as np

class EnhancedHybridEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(20, budget // 15)  # Reduced population size for efficiency
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(*self.bounds, (self.pop_size, dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.pop_size, dim))  # Adjusted velocity range
        self.personal_best_positions = self.population.copy()
        self.personal_best_values = np.full(self.pop_size, np.inf)
        self.global_best_position = self.population[np.random.choice(self.pop_size)]
        self.global_best_value = np.inf
        self.evals = 0
        self.F = 0.45  # Slightly reduced scaling factor for finer adjustments
        self.CR = 0.9  # Increased crossover rate for greater diversity
        self.omega = 0.25  # Further decreased inertia weight for faster convergence
        self.phi_p = 1.5  # Adjusted cognitive component for better local search
        self.phi_g = 1.3  # Adjusted social component for improved global exploration

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

            self.update_velocities()
            self.population += self.velocities
            np.clip(self.population, *self.bounds, out=self.population)

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

    def update_velocities(self):
        rand_vals = np.random.rand(2, self.pop_size, self.dim)
        cognitive_component = rand_vals[0] * self.phi_p * (self.personal_best_positions - self.population)
        social_component = rand_vals[1] * self.phi_g * (self.global_best_position - self.population)
        self.velocities = self.omega * self.velocities + cognitive_component + social_component