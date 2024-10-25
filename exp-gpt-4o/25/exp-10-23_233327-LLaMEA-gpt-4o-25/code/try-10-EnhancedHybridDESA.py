import numpy as np

class EnhancedHybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temperature = 1.0
        self.cooling_rate = 0.95

    def adapt_rates(self, evaluations):
        factor_decay = 0.95
        self.mutation_factor = max(0.5, self.mutation_factor * factor_decay)
        self.crossover_rate = min(1.0, self.crossover_rate * (1.0 + factor_decay * np.random.rand()))

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant_vector = np.clip(x1 + self.mutation_factor * (x2 - x3), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1
                
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

            self.temperature *= self.cooling_rate
            self.adapt_rates(evaluations)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]