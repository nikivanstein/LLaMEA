import numpy as np

class ImprovedHybridEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(40, budget // 10)  # Reduced population size for expediency
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(*self.bounds, (self.pop_size, dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.pop_size, dim))
        self.personal_best_positions = self.population.copy()
        self.personal_best_values = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evals = 0
        self.F = 0.6  # Adjusted to use dynamic scaling
        self.CR = 0.8  # Adjusted crossover rate
        self.omega = 0.4  # Decreased inertia weight
        self.phi_p = 1.5  # Adjusted cognitive component
        self.phi_g = 1.1  # Adjusted social component

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
        self.personal_best_positions = np.where(better_mask[:, np.newaxis], self.population, self.personal_best_positions)
        self.personal_best_values = np.where(better_mask, values, self.personal_best_values)
        if values.min() < self.global_best_value:
            self.global_best_value = values.min()
            self.global_best_position = self.population[values.argmin()]