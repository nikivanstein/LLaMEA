import numpy as np

class EvoQuantum:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def crowding_distance(self, population):
        distances = np.zeros(len(population))
        for i in range(self.dim):
            sorted_indices = np.argsort(population[:, i])
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf
            for j in range(1, len(population) - 1):
                distances[sorted_indices[j]] += population[sorted_indices[j+1], i] - population[sorted_indices[j-1], i]
        return distances
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness_values = np.array([func(individual) for individual in self.population])
            sorted_indices = np.argsort(fitness_values)
            elite = self.population[sorted_indices[:10]]
            new_population = np.tile(elite, (10, 1))
            
            theta = np.random.uniform(0, 2*np.pi, (self.budget, self.dim))
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)])
            new_population = np.tensordot(new_population, rotation_matrix, axes=([1], [2]))
            
            mutation_rate = 0.1
            mutation_mask = np.random.choice([0, 1], size=(self.budget, self.dim), p=[1 - mutation_rate, mutation_rate])
            new_population += mutation_mask * np.random.normal(0, 1, (self.budget, self.dim))
            
            crowding_distances = self.crowding_distance(new_population)
            sorted_crowding_indices = np.argsort(crowding_distances)[::-1]
            new_population = new_population[sorted_crowding_indices]
            
            self.population = new_population[:self.budget]
        best_solution = elite[0]
        return func(best_solution)