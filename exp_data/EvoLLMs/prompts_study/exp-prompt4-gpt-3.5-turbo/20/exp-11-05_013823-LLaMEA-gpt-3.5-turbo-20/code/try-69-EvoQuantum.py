import numpy as np

class EvoQuantum:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.crowding_distance = np.zeros(budget)  # Initialize crowding distance
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness_values = np.array([func(individual) for individual in self.population])
            sorted_indices = np.argsort(fitness_values)
            elite = self.population[sorted_indices[:10]]  # Select top 10% as elite
            
            # Calculate crowding distance
            for i in range(self.budget):
                self.crowding_distance[i] = np.sum(np.linalg.norm(self.population - self.population[i], axis=1))
            
            # Maintain diversity using crowding distance
            diversity_factor = np.argsort(self.crowding_distance)
            new_population = self.population[diversity_factor]  # Preserve diverse individuals
            
            new_population = np.tile(elite, (10, 1))  # Replicate elite 10 times
            
            # Apply quantum-inspired rotation gate
            theta = np.random.uniform(0, 2*np.pi, (self.budget, self.dim))
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)])
            new_population = np.tensordot(new_population, rotation_matrix, axes=([1], [2]))
            
            # Update population with mutation and crossover
            mutation_rate = 0.1
            mutation_mask = np.random.choice([0, 1], size=(self.budget, self.dim), p=[1 - mutation_rate, mutation_rate])
            new_population += mutation_mask * np.random.normal(0, 1, (self.budget, self.dim))
            
            self.population = new_population
        best_solution = elite[0]  # Select the best solution from the elite
        return func(best_solution)