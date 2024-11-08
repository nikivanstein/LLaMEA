import numpy as np

class EfficientMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(10 * self.dim, budget // 10)
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.F = 0.5
        self.CR = 0.9
        self.evaluations = 0

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                a, b, c = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant = np.clip(self.population[a] + self.F * (self.population[b] - self.population[c]), *self.bounds)
                crossover_points = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_points, mutant, self.population[i])
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
            if self.evaluations % (self.budget // 10) == 0:
                self.reduce_population()
        best_index = np.argmin(self.fitness)
        return self.population[best_index]

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
        
    def reduce_population(self):
        sorted_indices = np.argsort(self.fitness)
        self.population = self.population[sorted_indices[:self.population_size//2]]
        self.fitness = self.fitness[sorted_indices[:self.population_size//2]]
        self.population_size = len(self.population)
        new_population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        self.population = np.vstack((self.population, new_population))
        self.fitness = np.append(self.fitness, np.full(self.population_size, np.inf))