import numpy as np

class OptimizedDifferentialEvolution:
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

    def __call__(self, func):
        evaluations = 0
        indices = np.arange(self.population_size)
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                np.random.shuffle(indices)
                a, b, c = indices[:3]
                x_target = self.population[i]
                x1, x2, x3 = self.population[a], self.population[b], self.population[c]
                mutant_vector = np.clip(x1 + self.mutation_factor * (x2 - x3), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
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