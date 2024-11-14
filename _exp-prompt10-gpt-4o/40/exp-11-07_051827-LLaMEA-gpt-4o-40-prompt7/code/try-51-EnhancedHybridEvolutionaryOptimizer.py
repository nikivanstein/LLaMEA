import numpy as np

class EnhancedHybridEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)  # Increased population for diversity
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(*self.bounds, (self.pop_size, dim))
        self.velocities = np.zeros((self.pop_size, dim))  # Initial velocities set to zero
        self.personal_best_positions = self.population.copy()
        self.personal_best_values = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evals = 0
        self.F = 0.7  # Increased scaling factor for diversity
        self.CR = 0.9  # Higher crossover rate
        self.inertia_weight = 0.4  # Adjusted inertia weight
        self.cognitive_comp = 1.5  # Adjusted cognitive component
        self.social_comp = 1.3  # Adjusted social component

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
            self.velocities = (self.inertia_weight * self.velocities +
                               rand_vals[0] * self.cognitive_comp * (self.personal_best_positions - self.population) +
                               rand_vals[1] * self.social_comp * (self.global_best_position - self.population))
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