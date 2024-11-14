import numpy as np

class AdaptiveMemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(10 * dim, 50)  # Adjusted population size for efficiency
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.evals = 0
        self.mutation_factor = 0.5  # Fine-tuned mutation factor
        self.crossover_rate = 0.7  # Fine-tuned crossover rate
        self.strategy_switch_rate = 0.3  # Increased switch rate for more hybridization
        self.local_search_probability = 0.1  # New parameter for selective local search

    def __call__(self, func):
        while self.evals < self.budget:
            for i in range(self.pop_size):
                if self.evals >= self.budget:
                    break

                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)

                # Enhanced hybrid strategy switch
                if np.random.rand() < self.strategy_switch_rate:
                    d = self.population[np.random.choice(idxs)]
                    mutant = np.clip(a + self.mutation_factor * (b - c) + 0.5 * self.mutation_factor * (d - a), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                trial_fitness = func(trial)
                self.evals += 1

                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.population[i] = trial

                # Selective local search
                if np.random.rand() < self.local_search_probability:
                    trial = trial + np.random.normal(0, 0.1, self.dim)
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
        self.mutation_factor = 0.3 + np.random.rand() * 0.7
        self.crossover_rate = 0.5 + np.random.rand() * 0.5
        # More dynamic population size adaptation
        if np.random.rand() < 0.15:
            self.pop_size = max(4, min(int(self.pop_size * (0.8 + np.random.rand() * 0.4)), 50))