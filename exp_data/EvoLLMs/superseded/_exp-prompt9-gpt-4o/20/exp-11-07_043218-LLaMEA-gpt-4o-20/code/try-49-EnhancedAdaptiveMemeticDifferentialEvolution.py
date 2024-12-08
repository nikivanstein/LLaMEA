import numpy as np

class EnhancedAdaptiveMemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(12 * dim, 80)  # Increased population size for diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.evals = 0
        self.mutation_factor = 0.7  # Enhanced mutation factor for exploration
        self.crossover_rate = 0.9  # Enhanced crossover rate for better offspring
        self.strategy_switch_rate = 0.3  # Reduced to balance strategies
        self.local_search_probability = 0.2  # Intensified local search

    def __call__(self, func):
        while self.evals < self.budget:
            for i in range(self.pop_size):
                if self.evals >= self.budget:
                    break

                idxs = np.random.choice(self.pop_size, 4, replace=False)
                a, b, c, d = self.population[idxs]
                if np.random.rand() < self.strategy_switch_rate:
                    mutant = np.clip(a + self.mutation_factor * (b - c + d - a), self.lower_bound, self.upper_bound)
                else:
                    mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                trial_fitness = func(trial)
                self.evals += 1

                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.population[i] = trial

                if np.random.rand() < self.local_search_probability:
                    trial = trial + np.random.normal(0, 0.04, self.dim)  # Further reduced noise
                    trial = np.clip(trial, self.lower_bound, self.upper_bound)
                    trial_fitness = func(trial)
                    self.evals += 1
                    if trial_fitness < self.fitness[i]:
                        self.fitness[i] = trial_fitness
                        self.population[i] = trial

            self._adapt_parameters()
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]

    def _adapt_parameters(self):
        self.mutation_factor = 0.5 + np.random.rand() * 0.5  # Adjusted range
        self.crossover_rate = 0.7 + np.random.rand() * 0.3  # Adjusted range
        if np.random.rand() < 0.25:  # Adjusted probability
            self.pop_size = max(5, min(int(self.pop_size * (0.9 + np.random.rand() * 0.2)), 80))  # Updated range