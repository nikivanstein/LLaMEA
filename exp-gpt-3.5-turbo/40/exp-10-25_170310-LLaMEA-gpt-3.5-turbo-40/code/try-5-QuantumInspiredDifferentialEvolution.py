import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        F = 0.5  # Differential weight
        CR = 0.9  # Crossover probability
        
        for _ in range(self.budget):
            for i in range(self.budget):
                target = self.population[i]
                
                # Mutation
                a, b, c = np.random.choice(self.population, 3, replace=False)
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)
                
                # Quantum-inspired rotation
                angle = np.arctan(np.linalg.norm(mutant - target))
                rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                rotated_mutant = np.dot(rotation_matrix, mutant)
                
                # Crossover
                trial = np.where(np.random.rand(self.dim) < CR, rotated_mutant, target)
                
                # Selection
                if func(trial) < func(target):
                    self.population[i] = trial
        
        final_fitness = [func(x) for x in self.population]
        best_idx = np.argmin(final_fitness)
        best_solution = self.population[best_idx]
        
        return best_solution