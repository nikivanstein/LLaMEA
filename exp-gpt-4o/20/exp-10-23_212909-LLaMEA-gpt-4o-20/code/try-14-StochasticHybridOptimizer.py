import numpy as np

class StochasticHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.initial_population_size = 20 + int(2.0 * np.sqrt(dim))  # Slightly increased initial population size
        self.population_size = self.initial_population_size
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.evaluation_count = 0

    def __call__(self, func):
        best_solution = None
        best_score = np.inf
        
        F = 0.9  # Increased DE scaling factor for exploration
        CR = 0.85  # Adjusted crossover probability

        while self.evaluation_count < self.budget:
            for i in range(self.population_size):
                if self.scores[i] == np.inf:
                    self.scores[i] = func(self.population[i])
                    self.evaluation_count += 1
                    if self.scores[i] < best_score:
                        best_score = self.scores[i]
                        best_solution = self.population[i].copy()

            indices = np.arange(self.population_size)
            np.random.shuffle(indices)
            for i in indices:
                if self.evaluation_count >= self.budget:
                    break
                candidates = np.setdiff1d(indices, i)
                a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])
                trial = np.where(np.random.rand(self.dim) < CR, mutant, self.population[i])
                trial_score = func(trial)
                self.evaluation_count += 1
                if trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score
                    if trial_score < best_score:
                        best_score = trial_score
                        best_solution = trial
                else:
                    # Simulated annealing step
                    if np.random.rand() < np.exp(-(trial_score - self.scores[i]) / (1 + self.evaluation_count / self.budget)):
                        self.population[i] = trial
                        self.scores[i] = trial_score

            F = np.clip(F * (1 + np.random.uniform(-0.15, 0.15)), 0.5, 1.0)  # Broader adaptive F range
            CR = np.clip(CR + np.random.uniform(-0.1, 0.1), 0.6, 0.9)  # Broader adaptive CR range

            if (self.evaluation_count / self.budget) > 0.5:  # Earlier dynamic reduction threshold
                self.population_size = int(self.initial_population_size * 0.65)
                self.population = self.population[:self.population_size]
                self.scores = self.scores[:self.population_size]

        return best_solution, best_score