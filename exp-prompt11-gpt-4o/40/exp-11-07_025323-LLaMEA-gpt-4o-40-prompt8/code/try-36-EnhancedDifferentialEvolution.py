import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(max(5, 8 * dim), budget // 2)
        self.mutation_factor_base = 0.5
        self.crossover_rate = 0.85
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = self.population[np.random.randint(self.population_size)]
        self.best_value = np.inf

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                idx = np.random.choice(list(set(range(self.population_size)) - {i}), 3, replace=False)
                x_target = self.population[i]
                x1, x2, x3 = self.population[idx]
                adaptive_factor = self.mutation_factor_base + np.random.rand() * 0.3
                mutant_vector = np.clip(x1 + adaptive_factor * (x2 - x3), self.lower_bound, self.upper_bound)
                crossover_mask = (np.random.rand(self.dim) < self.crossover_rate)
                trial_vector = np.where(crossover_mask, mutant_vector, x_target)
                trial_value = func(trial_vector)
                evaluations += 1
                if trial_value < func(x_target):
                    self.population[i] = trial_vector
                    if trial_value < self.best_value:
                        self.best_solution = trial_vector
                        self.best_value = trial_value
        return self.best_solution