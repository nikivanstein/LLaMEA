import numpy as np

class EnhancedAdaptiveEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(20, budget // 10)  # Adjusted and slightly increased population size
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(*self.bounds, (self.pop_size, dim))
        self.personal_best_positions = self.population.copy()
        self.personal_best_values = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evals = 0
        self.F = 0.6  # Adaptive scaling factor
        self.CR = 0.8  # Adjusted crossover rate
        self.omega = 0.5  # Improved inertia weight for better exploration

    def __call__(self, func):
        self.evaluate_population(func)
        learning_rate = 0.1  # New adaptive learning rate

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

            self.update_population(func, learning_rate)

        return self.global_best_value

    def update_population(self, func, learning_rate):
        rand_vals = np.random.rand(self.pop_size, self.dim)
        for i in range(self.pop_size):
            cognitive_component = rand_vals[i] * (self.personal_best_positions[i] - self.population[i])
            social_component = rand_vals[i] * (self.global_best_position - self.population[i])
            self.population[i] += self.omega * (cognitive_component + social_component)
            self.population[i] = np.clip(self.population[i], *self.bounds)
            trial_value = func(self.population[i])
            self.evals += 1
            if trial_value < self.personal_best_values[i]:
                self.personal_best_positions[i] = self.population[i]
                self.personal_best_values[i] = trial_value
            if trial_value < self.global_best_value:
                self.global_best_value = trial_value
                self.global_best_position = self.population[i]

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