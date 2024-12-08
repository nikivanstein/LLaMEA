import numpy as np

class EDE_AMS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9 # Crossover probability

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def adaptive_mutation(self, xi, x1, x2, x3):
        strategy = np.random.choice(['rand', 'best'])
        if strategy == 'rand':
            return xi + self.F * (x1 - x2 + x3 - x2)
        else:
            return self.best_global_position + self.F * (x1 - x2)

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                x1, x2, x3 = self.population[np.random.choice(indices, 3, replace=False)]
                
                mutant = self.adaptive_mutation(self.population[i], x1, x2, x3)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.population[i])
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                
                trial_fitness = self.evaluate(func, trial)
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_global_fitness:
                        self.best_global_fitness = trial_fitness
                        self.best_global_position = trial

        return self.best_global_position