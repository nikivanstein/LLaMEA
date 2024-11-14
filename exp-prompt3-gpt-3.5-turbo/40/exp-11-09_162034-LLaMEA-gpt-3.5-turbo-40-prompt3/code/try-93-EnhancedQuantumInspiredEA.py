import numpy as np

class EnhancedQuantumInspiredEA(QuantumInspiredEA):
    def mutate(self, individual, mutation_rate=0.1):
        mutated_individual = individual + np.random.normal(0, 1, self.dim) * mutation_rate
        mutated_individual = np.clip(mutated_individual, -5.0, 5.0)
        return mutated_individual
        
    def evolve(self, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        elite = self.population[sorted_indices[0]]
        
        # Quantum rotation gate operation
        rotation_angle = np.arctan(np.sqrt(np.log(1 + np.arange(self.dim)) / np.arange(1, self.dim + 1)))
        for i in range(1, self.budget):
            for j in range(self.dim):
                if np.random.rand() < 0.5:
                    self.population[i][j] = np.cos(rotation_angle[j]) * self.population[i][j] - np.sin(rotation_angle[j]) * elite[j]
                else:
                    self.population[i][j] = np.sin(rotation_angle[j]) * self.population[i][j] + np.cos(rotation_angle[j]) * elite[j]
            
            # Integrate local search through mutation
            self.population[i] = self.mutate(self.population[i])

    def __call__(self, func):
        fitness_values = [func(ind) for ind in self.population]
        
        for _ in range(self.budget):
            self.evolve(fitness_values)
            fitness_values = [func(ind) for ind in self.population]
        
        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution