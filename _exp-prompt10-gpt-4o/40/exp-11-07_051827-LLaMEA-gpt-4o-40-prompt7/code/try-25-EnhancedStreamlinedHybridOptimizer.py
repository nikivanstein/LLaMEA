import numpy as np

class EnhancedStreamlinedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(40, budget // 12)
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(*self.bounds, (self.pop_size, dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.pop_size, dim))
        self.personal_best_positions = self.population.copy()
        self.personal_best_values = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evals = 0
        self.F = 0.5  
        self.CR = 0.75  
        self.omega = 0.5  
        self.phi_p = 1.4
        self.phi_g = 1.2

    def __call__(self, func):
        self.evaluate_population(func)

        while self.evals < self.budget:
            indices = np.random.randint(0, self.pop_size, (self.pop_size, 2))
            mutants = np.clip(self.population[indices[:, 0]] + self.F * 
                              (self.population[indices[:, 1]] - self.population[np.random.randint(0, self.pop_size, self.pop_size)]), *self.bounds)
            crossover_mask = np.random.rand(self.pop_size, self.dim) < self.CR
            trials = np.where(crossover_mask, mutants, self.population)

            trial_values = np.apply_along_axis(func, 1, trials)
            self.evals += self.pop_size

            improvements = trial_values < self.personal_best_values
            self.personal_best_positions = np.where(improvements[:, np.newaxis], trials, self.personal_best_positions)
            self.personal_best_values = np.where(improvements, trial_values, self.personal_best_values)
            
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
        self.personal_best_positions = np.where(better_mask[:, np.newaxis], self.population, self.personal_best_positions)
        self.personal_best_values = np.where(better_mask, values, self.personal_best_values)
        if values.min() < self.global_best_value:
            self.global_best_value = values.min()
            self.global_best_position = self.population[values.argmin()]