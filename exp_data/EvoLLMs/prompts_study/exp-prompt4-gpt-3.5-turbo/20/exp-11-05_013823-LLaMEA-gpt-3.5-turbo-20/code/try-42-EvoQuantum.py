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
            elite = self.population[sorted_indices[:10]]  # Select top 10% as elite
            new_population = np.tile(elite, (10, 1))  # Replicate elite 10 times
            
            # Enhanced quantum-inspired rotation gate
            thetas = np.random.uniform(0, 2*np.pi, (self.budget, self.dim))
            c_thetas = np.cos(thetas)
            s_thetas = np.sin(thetas)
            rotation_matrix = np.stack(((c_thetas, -s_thetas), (s_thetas, c_thetas)), axis=-1)
            new_population = np.dot(new_population, rotation_matrix)
            
            # Update population with mutation and crossover
            mutation_rate = 0.1
            mutation_mask = np.random.choice([0, 1], size=(self.budget, self.dim), p=[1 - mutation_rate, mutation_rate])
            new_population += mutation_mask * np.random.normal(0, 1, (self.budget, self.dim))
            
            self.population = new_population
        best_solution = elite[0]  # Select the best solution from the elite
        return func(best_solution)