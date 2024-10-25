import numpy as np

class EnhancedAdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.initial_population_size = 20 + int(2.5 * np.sqrt(dim))  # Adjusted initial population size for more diversity
        self.population_size = self.initial_population_size
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.evaluation_count = 0

    def __call__(self, func):
        best_solution = None
        best_score = np.inf
        
        F = 0.9  # Adjusted DE scaling factor for stronger mutation
        CR = 0.75  # Adjusted crossover probability for balance

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
                candidates = np.delete(indices, np.where(indices == i))
                a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]
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

            # Adaptive mutation and crossover rates
            F = np.clip(F * (1 + np.random.uniform(-0.1, 0.1)), 0.6, 1.2)  # Enhanced adaptive F for diversity
            CR = np.clip(CR + np.random.uniform(-0.05, 0.05), 0.65, 0.9)  # Enhanced adaptive CR for stability

            # Dynamic population adjustment
            if (self.evaluation_count / self.budget) > 0.8:  # Adjusted dynamic reduction threshold for late-stage focus
                new_population_size = int(self.initial_population_size * 0.5)
                if new_population_size < self.population_size:
                    self.population_size = new_population_size
                    self.population = self.population[:self.population_size]
                    self.scores = self.scores[:self.population_size]

        return best_solution, best_score