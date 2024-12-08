import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        best_solution = population[np.argmin(func(population))]
        
        for _ in range(self.budget):
            # Apply quantum rotation gates to population
            theta = np.random.uniform(-np.pi, np.pi, size=(self.dim))
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)])
            population = np.dot(population, rotation_matrix)
            
            # Apply differential evolution
            F = 0.5  # Differential weight
            for i in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + F * (b - c)
                trial = mutant + np.random.normal(0, 0.1, size=mutant.shape)
                if func(trial) < func(population[i]):
                    population[i] = trial

            current_best_solution = population[np.argmin(func(population))]
            if func(current_best_solution) < func(best_solution):
                best_solution = current_best_solution
        
        return best_solution