import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim, population_size=None):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size or 20 * dim
        self.population = np.random.uniform(-5, 5, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.F = 0.5  # initial mutation factor
        self.CR = 0.9  # initial crossover rate
        self.evaluations = 0
        self.prev_best_fitness = np.inf

    def __call__(self, func):
        best_idx = np.argmin(self.fitness)
        best = self.population[best_idx]
        
        while self.evaluations < self.budget:
            if self.evaluations % (self.budget // 10) == 0:  # Adjust population size dynamically
                self.population_size = min(int(self.population_size * 0.9), self.population_size)
                # Add momentum adaptation to F for smooth transitions
                if self.prev_best_fitness < self.fitness[best_idx]:
                    self.F = np.clip(self.F * 0.95, 0.4, 0.9)
                self.prev_best_fitness = self.fitness[best_idx]

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), -5, 5)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, self.population[i])
                
                trial_fitness = func(trial)
                self.evaluations += 1
                
                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.population[i] = trial
                    if trial_fitness < self.fitness[best_idx]:
                        best = trial
                        best_idx = i
                        # Slightly adjust F and CR for better exploration and exploitation
                        self.F = np.clip(self.F + 0.01 * (np.random.rand() - 0.5), 0.4, 0.9)
                        self.CR = np.clip(self.CR + 0.01 * (np.random.rand() - 0.5), 0.8, 1.0)

        return best