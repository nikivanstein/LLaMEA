import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.f = 0.5  # scaling factor
        self.cr = 0.9  # crossover probability
        self.strategy_switch_ratio = 0.5  # ratio to switch between exploration and exploitation

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        num_evaluations = self.pop_size

        while num_evaluations < self.budget:
            # Vectorized mutation and crossover
            idx = np.arange(self.pop_size)
            np.random.shuffle(idx)
            a, b, c = idx[:3]
            mutant_vectors = population[a] + self.f * (population[b] - population[c])
            mutant_vectors = np.clip(mutant_vectors, self.lower_bound, self.upper_bound)

            crossover_mask = np.random.rand(self.pop_size, self.dim) < self.cr
            trials = np.where(crossover_mask, mutant_vectors, population)

            # Vectorized fitness evaluation
            trial_fitness = np.apply_along_axis(func, 1, trials)
            num_evaluations += self.pop_size

            # Vectorized selection
            improved = trial_fitness < fitness
            population[improved] = trials[improved]
            fitness[improved] = trial_fitness[improved]

            # Adaptive mutation strategy based on evaluations
            if num_evaluations / self.budget > self.strategy_switch_ratio:
                self.f = np.random.uniform(0.4, 0.9)  # switch to more exploitation

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]