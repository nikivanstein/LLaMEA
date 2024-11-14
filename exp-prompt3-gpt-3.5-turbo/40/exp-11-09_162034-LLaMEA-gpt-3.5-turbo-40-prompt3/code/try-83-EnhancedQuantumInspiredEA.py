import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def differential_evolution(self, fitness_values):
        F = 0.5
        for i in range(1, self.budget):
            r1, r2, r3 = np.random.choice(self.budget, 3, replace=False)
            mutant = self.population[r1] + F * (self.population[r2] - self.population[r3])
            crossover_points = np.random.rand(self.dim) < 0.5
            trial = np.where(crossover_points, mutant, self.population[i])
            if func(trial) < fitness_values[i]:
                self.population[i] = trial
    
    def __call__(self, func):
        fitness_values = [func(ind) for ind in self.population]
        
        for _ in range(self.budget):
            self.evolve(fitness_values)
            self.differential_evolution(fitness_values)
            fitness_values = [func(ind) for ind in self.population]
        
        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution