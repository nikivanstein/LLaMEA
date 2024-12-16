import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 10 * dim
        self.F = np.random.uniform(0.4, 0.9)
        self.CR = np.random.uniform(0.8, 1.0)
        self.population = None
        self.best_individual = None
        self.best_value = float('inf')
        
    def initialize_population(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.evaluate_population()
    
    def evaluate_population(self):
        fitness = np.apply_along_axis(func, 1, self.population)
        best_idx = np.argmin(fitness)
        self.best_value = fitness[best_idx]
        self.best_individual = self.population[best_idx]
        
    def mutate(self, idx):
        indices = np.delete(np.arange(self.population_size), idx)
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        mutant = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
        return np.clip(mutant, self.lb, self.ub)
    
    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial
    
    def select(self, target_idx, trial, func):
        trial_value = func(trial)
        if trial_value < self.best_value:
            self.best_value = trial_value
            self.best_individual = trial
        if trial_value < func(self.population[target_idx]):
            self.population[target_idx] = trial
    
    def adaptive_parameters(self, iteration, max_iterations):
        self.F = 0.5 + 0.5 * (1 - iteration / max_iterations)
        self.CR = 0.9 - 0.5 * (iteration / max_iterations)

    def __call__(self, func):
        self.initialize_population()
        evals = 0
        iteration = 0
        max_iterations = self.budget // self.population_size
        while evals < self.budget:
            for idx in range(self.population_size):
                if evals >= self.budget:
                    break
                self.adaptive_parameters(iteration, max_iterations)
                target = self.population[idx]
                mutant = self.mutate(idx)
                trial = self.crossover(target, mutant)
                self.select(idx, trial, func)
                evals += 1
            iteration += 1
        return self.best_individual, self.best_value