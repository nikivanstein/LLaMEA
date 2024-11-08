import numpy as np

class EfficientMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 8 * self.dim  # Reduced population size for efficiency
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(*self.bounds, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.F = 0.7  # Adjusted mutation factor for better exploration
        self.CR = 0.8  # Adjusted crossover rate
        self.evaluations = 0

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                # Mutation with enhanced selection strategy
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, *self.bounds)
                
                # Crossover with adaptive strategy
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                # Adaptive F and CR based on fitness improvement
                if trial_fitness < self.fitness.mean():
                    self.F = np.clip(self.F + 0.02 * np.random.randn(), 0.1, 1.0)
                    self.CR = np.clip(self.CR + 0.02 * np.random.randn(), 0.1, 1.0)

        # Return the best solution found
        return self.population[np.argmin(self.fitness)]

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.evaluations >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1