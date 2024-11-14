import numpy as np

class ANEDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(4, int(0.5 * self.budget / self.dim))
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.best_solution = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation with adjusted dynamic scaling factor
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                dynamic_F = self.F * (1 + np.random.uniform(-0.15, 0.15))
                mutant = np.clip(a + dynamic_F * (b - c), self.lower_bound, self.upper_bound)

                # Adaptive Crossover
                adaptive_CR = self.CR + 0.05 * np.random.uniform(-1, 1)
                trial = np.copy(population[i])
                crossover_mask = np.random.rand(self.dim) < adaptive_CR
                trial[crossover_mask] = mutant[crossover_mask]
                
                # Local attractor with refined exploration
                local_center = np.median(population, axis=0)
                trial = np.clip(trial + np.random.uniform(-0.1, 0.1, self.dim) * (local_center - trial), self.lower_bound, self.upper_bound)

                # Probabilistic local search
                if np.random.rand() < 0.15:  # New probabilistic factor
                    local_search_offset = np.random.normal(0, 0.1, self.dim)
                    trial = np.clip(trial + local_search_offset, self.lower_bound, self.upper_bound)

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Update global best
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

        return self.best_solution, self.best_fitness