import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def levy_flight(self, x):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        s = np.random.normal(0, sigma, size=len(x))
        u = np.random.normal(0, 1, size=len(x))
        v = np.random.normal(0, 1, size=len(x))
        step = s / (abs(u) ** (1 / beta))
        step /= np.linalg.norm(step)
        x = x + 0.01 * step * v
        return x
    
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
            
            # Levy flight mutation
            self.population[i] = self.levy_flight(self.population[i])
    
    def __call__(self, func):
        fitness_values = [func(ind) for ind in self.population]
        
        for _ in range(self.budget):
            self.evolve(fitness_values)
            fitness_values = [func(ind) for ind in self.population]
        
        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution