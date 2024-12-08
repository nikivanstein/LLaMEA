import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def evolve(self, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        elite = self.population[sorted_indices[0]]

        # Differential Evolution Strategy
        mutation_factor = 0.5
        crossover_prob = 0.7
        for i in range(1, self.budget):
            idxs = [idx for idx in range(self.budget) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mutation_factor * (b - c), -5.0, 5.0)
            crossover_mask = np.random.rand(self.dim) < crossover_prob
            self.population[i] = np.where(crossover_mask, mutant, self.population[i])
    
    def __call__(self, func):
        fitness_values = [func(ind) for ind in self.population]
        
        for _ in range(self.budget):
            self.evolve(fitness_values)
            fitness_values = [func(ind) for ind in self.population]
        
        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution