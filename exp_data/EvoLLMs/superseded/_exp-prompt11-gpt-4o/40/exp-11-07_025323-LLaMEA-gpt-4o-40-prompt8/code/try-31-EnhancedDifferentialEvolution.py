import numpy as np
from joblib import Parallel, delayed

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
        self.function_values = np.array([np.inf] * self.population_size)
        self.best_solution = np.zeros(dim)
        self.best_value = np.inf

    def __call__(self, func):
        evaluations = 0
        def evaluate_trial(trial_vector):
            return func(trial_vector)
        
        while evaluations < self.budget:
            indices = np.arange(self.population_size)
            np.random.shuffle(indices)
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                idx = np.random.choice(indices[indices != i], 3, replace=False)
                x_target = self.population[i]
                x1, x2, x3 = self.population[idx[0]], self.population[idx[1]], self.population[idx[2]]
                adaptive_factor = self.mutation_factor_base + np.random.rand() * 0.3
                mutant_vector = x1 + adaptive_factor * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                crossover_mask = (np.random.rand(self.dim) < self.crossover_rate)
                trial_vector = np.where(crossover_mask, mutant_vector, x_target)
                
                trial_value = Parallel(n_jobs=-1)(delayed(evaluate_trial)(trial_vector))[0]
                evaluations += 1
                
                if trial_value < self.function_values[i]:
                    self.population[i] = trial_vector
                    self.function_values[i] = trial_value
                    if trial_value < self.best_value:
                        self.best_solution = trial_vector.copy()
                        self.best_value = trial_value
        return self.best_solution