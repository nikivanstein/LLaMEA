import numpy as np

class Enhanced_DE_SA_Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.CR = 0.5
        self.F = 0.5
        self.T = 1.0
        self.alpha = 0.9
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        
    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            mutant_indices = np.random.randint(self.pop_size, size=(2, self.pop_size))
            mutant = self.population[mutant_indices]
            trial = self.population + self.F * (mutant[0] - mutant[1])
            mask = np.random.rand(self.pop_size, self.dim) < self.CR
            trial[mask] = self.population[mask]
            
            new_fitness = np.array([func(individual) for individual in trial])
            improve_mask = new_fitness < best_fitness
            best_solution[improve_mask] = trial[improve_mask]
            best_fitness[improve_mask] = new_fitness[improve_mask]
            
            random_mask = np.random.rand(self.pop_size) < np.exp((func(self.population) - new_fitness) / self.T)
            self.population = np.where(random_mask[:, None], trial, self.population)
            self.T = np.maximum(self.alpha * self.T, 0.1)
            
        return best_solution