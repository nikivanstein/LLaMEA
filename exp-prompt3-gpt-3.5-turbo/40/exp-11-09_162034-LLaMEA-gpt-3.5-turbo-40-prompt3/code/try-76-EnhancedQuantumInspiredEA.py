import numpy as np

class EnhancedQuantumInspiredEA(QuantumInspiredEA):
    def evolve(self, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        elite = self.population[sorted_indices[0]]
        
        # Adaptive mutation rates based on fitness
        mutation_rates = 1 / np.sqrt(np.arange(1, self.dim + 1))
        for i in range(1, self.budget):
            for j in range(self.dim):
                mutation_rate = mutation_rates[j]
                if np.random.rand() < 0.5:
                    self.population[i][j] += mutation_rate * (elite[j] - self.population[i][j])
                else:
                    self.population[i][j] -= mutation_rate * (elite[j] - self.population[i][j])
    
    def __call__(self, func):
        fitness_values = [func(ind) for ind in self.population]
        
        for _ in range(self.budget):
            self.evolve(fitness_values)
            fitness_values = [func(ind) for ind in self.population]
        
        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution