import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(12 * dim, 60)  # Adjusted population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.evals = 0
        self.mutation_factor = 0.6  # Adjusted mutation factor for exploration
        self.crossover_rate = 0.6  # Adjusted crossover rate
        self.strategy_switch_rate = 0.2  # New parameter for hybrid mutation strategy

    def __call__(self, func):
        while self.evals < self.budget:
            for i in range(self.pop_size):
                if self.evals >= self.budget:
                    break

                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                
                # Hybrid strategy switch
                if np.random.rand() < self.strategy_switch_rate:
                    d = self.population[np.random.choice(idxs)]
                    mutant = np.clip(a + self.mutation_factor * (b - c) + self.mutation_factor * (d - a), self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                trial_fitness = func(trial)
                self.evals += 1

                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.population[i] = trial

            self._adapt_parameters()
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]

    def _adapt_parameters(self):
        self.mutation_factor = 0.4 + np.random.rand() * 0.6
        self.crossover_rate = 0.4 + np.random.rand() * 0.6
        if np.random.rand() < 0.1:  # Population size adaptation
            self.pop_size = max(4, min(int(self.pop_size * (0.9 + np.random.rand() * 0.2)), 60))