import numpy as np

class EvoQuantum:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness_values = np.array([func(individual) for individual in self.population])
            sorted_indices = np.argsort(fitness_values)
            elite = self.population[sorted_indices[:int(0.1*self.budget)]]  # Select top 10% as elite
            new_population = np.tile(elite, (10, 1))  # Replicate elite 10 times
            
            # Apply quantum-inspired rotation gate
            thetas = np.random.uniform(0, 2*np.pi, (self.budget, self.dim))
            rotation_matrix = np.array([[np.cos(thetas.flatten()), -np.sin(thetas.flatten())], [np.sin(thetas.flatten()), np.cos(thetas.flatten())]]).reshape(self.budget, 2, 2)
            new_population = np.einsum('ijk,ik->ij', rotation_matrix, new_population)
            
            # Update population with mutation and crossover
            mutation_rate = 0.1
            mutation_mask = np.random.choice([0, 1], size=(self.budget, self.dim), p=[1 - mutation_rate, mutation_rate])
            new_population += mutation_mask * np.random.normal(0, 1, (self.budget, self.dim))
            
            self.population = new_population
        best_solution = elite[0]  # Select the best solution from the elite
        return func(best_solution)