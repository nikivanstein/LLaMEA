import numpy as np

class AdaptiveGradientMutationOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.initial_population_size = 20 + int(2.0 * np.sqrt(dim))
        self.population_size = self.initial_population_size
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.evaluation_count = 0

    def __call__(self, func):
        best_solution = None
        best_score = np.inf
        
        F = 0.9  # Set DE scaling factor slightly higher
        CR = 0.7  # Set crossover probability slightly lower

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
                # Mutation using gradient-based approach
                candidates = np.setdiff1d(indices, i)
                a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]
                grad_mutant = np.clip(a + F * (b - c) + 0.1 * np.sign(b - c) * (self.bounds[1] - self.bounds[0]), self.bounds[0], self.bounds[1])
                # Crossover
                trial = np.where(np.random.rand(self.dim) < CR, grad_mutant, self.population[i])
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
            F = np.clip(F * (1 + np.random.uniform(-0.1, 0.1)), 0.5, 1.2)  # Adapt F with wider range
            CR = np.clip(CR + np.random.uniform(-0.05, 0.05), 0.5, 0.9)  # Adapt CR with smaller change

            # Dynamic population adjustment
            if (self.evaluation_count / self.budget) > 0.8:  # Change adjustment threshold
                new_population_size = max(int(self.initial_population_size * 0.5), 5)  # Ensure size doesn't get too small
                if new_population_size < self.population_size:
                    self.population_size = new_population_size
                    self.population = self.population[:self.population_size]
                    self.scores = self.scores[:self.population_size]

        return best_solution, best_score