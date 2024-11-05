import numpy as np

class ImprovedEvoQuantum:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def crowding_distance_selection(self, population, fitness_values, n_select):
        sorted_indices = np.argsort(fitness_values)
        crowding_distance = np.zeros(len(population))
        crowding_distance[sorted_indices[0]] = np.inf
        crowding_distance[sorted_indices[-1]] = np.inf
        for i in range(1, len(sorted_indices) - 1):
            crowding_distance[sorted_indices[i]] += fitness_values[sorted_indices[i+1]] - fitness_values[sorted_indices[i-1]]
        selected_indices = np.argsort(-crowding_distance)[:n_select]
        return population[selected_indices]

    def __call__(self, func):
        for _ in range(self.budget):
            fitness_values = np.array([func(individual) for individual in self.population])
            elite = self.crowding_distance_selection(self.population, fitness_values, int(self.budget * 0.1))  # Select top 10% based on crowding distance
            new_population = np.tile(elite, (10, 1))  # Replicate elite 10 times
            
            theta = np.random.uniform(0, 2*np.pi, (self.budget, self.dim))
            rotation_matrix = np.stack([np.stack([np.cos(theta), -np.sin(theta)], axis=-1), np.stack([np.sin(theta), np.cos(theta)], axis=-1)], axis=1)
            new_population = np.matmul(new_population, rotation_matrix)
            
            mutation_rate = 0.1
            mutation_mask = np.random.choice([0, 1], size=(self.budget, self.dim), p=[1 - mutation_rate, mutation_rate])
            new_population += mutation_mask * np.random.normal(0, 1, (self.budget, self.dim))
            
            self.population = new_population
        best_solution = elite[0]
        return func(best_solution)