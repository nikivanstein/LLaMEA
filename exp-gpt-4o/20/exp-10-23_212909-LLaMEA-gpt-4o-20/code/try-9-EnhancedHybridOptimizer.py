import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.initial_population_size = 10 + int(2 * np.sqrt(dim))
        self.population_size = self.initial_population_size
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.evaluation_count = 0
        self.memory = np.full(self.dim, np.nan)

    def __call__(self, func):
        best_solution = None
        best_score = np.inf
        
        F = 0.7  # Differential evolution scaling factor
        CR = 0.85  # Crossover probability

        while self.evaluation_count < self.budget:
            # Evaluate current population
            for i in range(self.population_size):
                if self.scores[i] == np.inf:
                    self.scores[i] = func(self.population[i])
                    self.evaluation_count += 1
                    if self.scores[i] < best_score:
                        best_score = self.scores[i]
                        best_solution = self.population[i].copy()

            # Evolutionary selection and differential evolution operation
            indices = np.arange(self.population_size)
            np.random.shuffle(indices)
            for i in indices:
                if self.evaluation_count >= self.budget:
                    break
                # Mutation
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])
                # Crossover
                trial = np.where(np.random.rand(self.dim) < CR, mutant, self.population[i])
                trial_score = func(trial)
                self.evaluation_count += 1
                # Selection
                if trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score
                    if trial_score < best_score:
                        best_score = trial_score
                        best_solution = trial
                        self.memory = trial.copy()

            # Adaptive mutation and crossover rates
            F = max(0.4, F * (1 + np.random.uniform(-0.15, 0.15)))
            CR = np.clip(CR + np.random.uniform(-0.05, 0.05), 0.65, 0.9)

            # Dynamic population adjustment
            if (self.evaluation_count / self.budget) > 0.5:
                self.population_size = int(self.initial_population_size * 0.75)
                self.population = self.population[:self.population_size]
                self.scores = self.scores[:self.population_size]

            # Incorporate memory for learning
            if np.random.rand() < 0.1 and not np.isnan(self.memory).any():
                idx = np.random.choice(self.population_size)
                perturbation = np.random.randn(self.dim) * 0.1
                self.population[idx] = np.clip(self.memory + perturbation, self.bounds[0], self.bounds[1])

        return best_solution, best_score