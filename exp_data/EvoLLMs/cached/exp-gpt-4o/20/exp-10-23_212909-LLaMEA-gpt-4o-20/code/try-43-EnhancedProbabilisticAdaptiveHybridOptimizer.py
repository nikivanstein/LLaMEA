import numpy as np

class EnhancedProbabilisticAdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.initial_population_size = 20 + int(1.5 * np.sqrt(dim))  # Slightly reduced initial population size
        self.population_size = self.initial_population_size
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.evaluation_count = 0

    def __call__(self, func):
        best_solution = None
        best_score = np.inf

        F = 0.9  # Slightly higher DE scaling factor
        CR = 0.75  # Slightly reduced crossover probability

        while self.evaluation_count < self.budget:
            # Evaluate current population
            for i in range(self.population_size):
                if self.scores[i] == np.inf:
                    self.scores[i] = func(self.population[i])
                    self.evaluation_count += 1
                    if self.scores[i] < best_score:
                        best_score = self.scores[i]
                        best_solution = self.population[i].copy()

            # Stochastic tournament selection and differential evolution operation
            indices = np.arange(self.population_size)
            np.random.shuffle(indices)
            for i in indices:
                if self.evaluation_count >= self.budget:
                    break
                # Tournament selection for mutation
                candidates = np.random.choice(indices, 5, replace=False)
                candidates_scores = self.scores[candidates]
                winner = candidates[np.argmin(candidates_scores)]
                other_candidates = np.setdiff1d(candidates, winner)
                a, b = self.population[np.random.choice(other_candidates, 2, replace=False)]
                mutant = np.clip(winner + F * (a - b), self.bounds[0], self.bounds[1])
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

            # Reinforced adaptive mutation and crossover rates
            F = np.clip(F * (1 + np.random.uniform(-0.2, 0.2)), 0.5, 1.2)  # Expanded adaptive F range
            CR = np.clip(CR + np.random.uniform(-0.1, 0.1), 0.5, 0.9)  # Expanded adaptive CR range

            # Dynamic population adjustment
            if (self.evaluation_count / self.budget) > 0.8:  # Later dynamic reduction threshold
                new_population_size = int(self.initial_population_size * 0.5)
                if new_population_size < self.population_size:
                    self.population_size = new_population_size
                    self.population = self.population[:self.population_size]
                    self.scores = self.scores[:self.population_size]

        return best_solution, best_score