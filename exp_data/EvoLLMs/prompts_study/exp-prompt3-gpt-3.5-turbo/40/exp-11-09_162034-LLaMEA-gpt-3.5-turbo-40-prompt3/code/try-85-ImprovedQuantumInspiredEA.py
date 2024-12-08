import numpy as np

class ImprovedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def evolve(self, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        elite = self.population[sorted_indices[0]]
        
        for i in range(1, self.budget):
            random_indices = np.random.choice(self.budget, 3, replace=False)
            mutant = self.population[random_indices[0]] + 0.5 * (self.population[random_indices[1]] - self.population[random_indices[2]])
            trial = np.where(np.random.rand(dim) < 0.5, mutant, self.population[i])
            
            if func(trial) < fitness_values[i]:
                self.population[i] = trial
    
    def __call__(self, func):
        fitness_values = [func(ind) for ind in self.population]
        
        for _ in range(self.budget):
            self.evolve(fitness_values)
            fitness_values = [func(ind) for ind in self.population]
        
        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution