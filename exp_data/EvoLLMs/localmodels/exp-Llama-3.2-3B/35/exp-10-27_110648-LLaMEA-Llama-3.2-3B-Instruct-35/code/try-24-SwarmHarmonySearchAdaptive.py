import numpy as np
import random

class SwarmHarmonySearchAdaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 50
        self.harmony_size = 20
        self.pheromone_matrix = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.best_solution = None
        self.fitness_values = np.zeros(self.population_size)
        self.mutation_prob = 0.35

    def __call__(self, func):
        for _ in range(self.budget):
            # Initialize the population with random solutions
            population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))

            # Evaluate the fitness of each solution
            self.fitness_values = func(population)

            # Update the pheromone matrix
            for i in range(self.population_size):
                self.pheromone_matrix[i] = self.fitness_values[i] / np.sum(self.fitness_values)

            # Select the best solutions
            indices = np.argsort(self.fitness_values)
            self.best_solution = population[indices[:self.harmony_size]]

            # Perform the harmony search
            for i in range(self.population_size):
                # Calculate the harmony memory
                harmony_memory = self.pheromone_matrix[indices[:self.harmony_size]]
                # Calculate the average harmony
                average_harmony = np.mean(harmony_memory, axis=0)
                # Calculate the difference between the current solution and the average harmony
                difference = population[i] - average_harmony
                # Update the current solution
                population[i] = population[i] + self.bounds[1] - self.bounds[0] * difference

            # Evaluate the fitness of the new solutions
            self.fitness_values = func(population)

            # Update the pheromone matrix
            for i in range(self.population_size):
                self.pheromone_matrix[i] = self.fitness_values[i] / np.sum(self.fitness_values)

            # Check if the best solution has improved
            if self.fitness_values[indices[0]] < self.fitness_values[indices[1]]:
                self.best_solution = population[indices[0]]

            # Refine the strategy with adaptive mutation
            refine_indices = np.random.choice(self.population_size, int(self.population_size * self.mutation_prob), replace=False)
            for index in refine_indices:
                # Generate a new individual
                new_individual = population[index] + np.random.uniform(-self.bounds[1] + self.bounds[0], self.bounds[1] - self.bounds[0])
                # Evaluate the fitness of the new individual
                new_fitness = func(new_individual)
                # Check if the new individual is better
                if new_fitness < self.fitness_values[index]:
                    population[index] = new_individual
                    self.fitness_values[index] = new_fitness

# Example usage
def func(x):
    return np.sum(x**2)

swarm_harmony_search_adaptive = SwarmHarmonySearchAdaptive(100, 5)
best_solution = swarm_harmony_search_adaptive(func)
print("Best solution:", best_solution)
print("Fitness value:", func(best_solution))