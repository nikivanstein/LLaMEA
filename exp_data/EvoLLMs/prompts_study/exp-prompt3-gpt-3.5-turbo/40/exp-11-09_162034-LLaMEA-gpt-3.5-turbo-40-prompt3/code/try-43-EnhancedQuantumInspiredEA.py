import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def differential_evolution(self, fitness_values, f=0.5, cr=0.9):
        for i in range(1, self.budget):
            candidate = self.population[i].copy()
            r1, r2, r3 = np.random.choice(np.delete(np.arange(self.budget), i), 3, replace=False)
            mutant = self.population[r1] + f * (self.population[r2] - self.population[r3])
            crossover_points = np.random.rand(self.dim) < cr
            trial = np.where(crossover_points, mutant, candidate)
            
            if fitness_values[i] < fitness_values[np.argmin([fitness_values[i], func(candidate)])]:
                self.population[i] = trial
    
    def __call__(self, func):
        fitness_values = [func(ind) for ind in self.population]
        
        for _ in range(self.budget):
            self.evolve(fitness_values)
            self.differential_evolution(fitness_values)
            fitness_values = [func(ind) for ind in self.population]
        
        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution