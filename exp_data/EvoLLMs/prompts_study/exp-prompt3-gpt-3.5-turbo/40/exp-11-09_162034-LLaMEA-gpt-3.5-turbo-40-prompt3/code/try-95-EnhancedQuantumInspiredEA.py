import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def evolve(self, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        elite = self.population[sorted_indices[0]]
        
        # Differential evolution with crossover
        F = 0.5  # Weight factor for differential evolution
        CR = 0.9  # Crossover probability
        for i in range(1, self.budget):
            random_indices = np.random.choice(range(self.budget), 3, replace=False)
            trial_vector = self.population[random_indices[0]] + F * (self.population[random_indices[1]] - self.population[random_indices[2]])
            crossover_mask = np.random.rand(self.dim) < CR
            self.population[i] = np.where(crossover_mask, trial_vector, self.population[i])
    
    def __call__(self, func):
        fitness_values = [func(ind) for ind in self.population]
        
        for _ in range(self.budget):
            self.evolve(fitness_values)
            fitness_values = [func(ind) for ind in self.population]
        
        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution