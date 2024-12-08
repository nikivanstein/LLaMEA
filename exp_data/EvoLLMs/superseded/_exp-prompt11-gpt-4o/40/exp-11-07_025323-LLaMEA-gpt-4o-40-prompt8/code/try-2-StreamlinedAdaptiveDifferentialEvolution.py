import numpy as np

class StreamlinedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 10 * dim)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.function_values = np.array([np.inf] * self.population_size)
        self.best_solution = None
        self.best_value = np.inf
        self.crossover_probability = np.random.rand(self.population_size, dim)

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                x_target = self.population[i]
                indices = np.random.choice(self.population_size, 4, replace=False)
                x1, x2, x3, x4 = self.population[indices]
                mutant_vector = x1 + self.mutation_factor * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                crossover_mask = self.crossover_probability[i] < self.crossover_rate
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(self.dim)] = True
                trial_vector = np.where(crossover_mask, mutant_vector, x_target)
                trial_value = func(trial_vector)
                evaluations += 1
                if trial_value < self.function_values[i]:
                    self.population[i] = trial_vector
                    self.function_values[i] = trial_value
                    if trial_value < self.best_value:
                        self.best_solution = trial_vector
                        self.best_value = trial_value
        return self.best_solution