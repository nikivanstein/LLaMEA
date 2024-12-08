import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            
            # Dynamic Quantum-inspired update rule based on fitness
            for i in range(self.budget):
                angle = np.arctan(np.linalg.norm(self.population[i] - best_solution) / (np.mean(fitness) + 1e-6))
                rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                self.population[i] = np.dot(rotation_matrix, self.population[i])
        
        final_fitness = [func(x) for x in self.population]
        best_idx = np.argmin(final_fitness)
        best_solution = self.population[best_idx]
        
        return best_solution