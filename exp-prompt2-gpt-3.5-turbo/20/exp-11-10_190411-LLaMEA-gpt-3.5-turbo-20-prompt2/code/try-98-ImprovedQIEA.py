import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        best_solution = population[np.argmin(func(population))]
        
        for _ in range(self.budget):
            # Apply enhanced quantum rotation gates to population
            theta = np.random.uniform(-np.pi, np.pi, size=(self.dim))
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            population = np.dot(population, rotation_matrix)
            
            # Apply differential evolution
            F = 0.5  # Differential weight
            for i in range(self.budget):
                idxs = np.random.choice(np.setdiff1d(range(self.budget), i, assume_unique=True), size=2, replace=False)
                mutant = population[idxs[0]] + F * (population[idxs[1]] - population[i])
                trial = mutant + np.random.normal(0, 0.1, size=self.dim)
                if func(trial) < func(population[i]):
                    population[i] = trial
                    
            current_best_solution = population[np.argmin(func(population))]
            if func(current_best_solution) < func(best_solution):
                best_solution = current_best_solution
        
        return best_solution