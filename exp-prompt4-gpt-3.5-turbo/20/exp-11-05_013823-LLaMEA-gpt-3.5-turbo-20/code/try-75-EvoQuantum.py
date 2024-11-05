import numpy as np

class EvoQuantum:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        temperature = 1.0
        cooling_rate = 0.003
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        
        for _ in range(self.budget):
            fitness_values = np.array([func(individual) for individual in self.population])
            sorted_indices = np.argsort(fitness_values)
            elite = self.population[sorted_indices[:10]]  # Select top 10% as elite
            new_population = np.tile(elite, (10, 1))  # Replicate elite 10 times
            
            # Apply quantum-inspired rotation gate
            theta = np.random.uniform(0, 2*np.pi, (self.budget, self.dim))
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)])
            new_population = np.tensordot(new_population, rotation_matrix, axes=([1], [2]))
            
            # Update population with simulated annealing
            for i in range(self.budget):
                candidate_solution = self.population[i]
                new_solution = candidate_solution + np.random.normal(0, 1, self.dim)
                cost_diff = func(new_solution) - func(candidate_solution)
                if cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / temperature):
                    self.population[i] = new_solution
            
            if func(self.population[0]) < func(best_solution):
                best_solution = self.population[0]
            
            temperature *= 1 - cooling_rate
        
        return func(best_solution)