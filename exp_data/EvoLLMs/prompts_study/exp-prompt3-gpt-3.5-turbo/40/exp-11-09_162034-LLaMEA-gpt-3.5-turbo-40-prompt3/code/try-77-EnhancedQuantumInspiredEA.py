import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def local_search(self, individual, func):
        current_fitness = func(individual)
        epsilon = 1e-6
        for _ in range(5):
            perturbation = np.random.uniform(-epsilon, epsilon, self.dim)
            new_individual = individual + perturbation
            new_fitness = func(new_individual)
            if new_fitness < current_fitness:
                individual = new_individual
                current_fitness = new_fitness
        return individual
    
    def evolve(self, fitness_values, func):
        sorted_indices = np.argsort(fitness_values)
        elite = self.population[sorted_indices[0]]
        
        rotation_angle = np.arctan(np.sqrt(np.log(1 + np.arange(self.dim)) / np.arange(1, self.dim + 1)))
        for i in range(1, self.budget):
            for j in range(self.dim):
                if np.random.rand() < 0.5:
                    self.population[i][j] = np.cos(rotation_angle[j]) * self.population[i][j] - np.sin(rotation_angle[j]) * elite[j]
                else:
                    self.population[i][j] = np.sin(rotation_angle[j]) * self.population[i][j] + np.cos(rotation_angle[j]) * elite[j]
            self.population[i] = self.local_search(self.population[i], func)
    
    def __call__(self, func):
        fitness_values = [func(ind) for ind in self.population]
        
        for _ in range(self.budget):
            self.evolve(fitness_values, func)
            fitness_values = [func(ind) for ind in self.population]
        
        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution