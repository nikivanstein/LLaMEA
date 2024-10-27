import numpy as np
import random

class SwarmHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 50
        self.harmony_size = 20
        self.pheromone_matrix = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.best_solution = None
        self.fitness_values = np.zeros(self.population_size)
        self.refinement_probability = 0.35

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

            # Refine the best solution with probability 0.35
            if random.random() < self.refinement_probability:
                # Randomly select two individuals from the best solutions
                individual1 = self.best_solution[np.random.choice(self.harmony_size)]
                individual2 = self.best_solution[np.random.choice(self.harmony_size)]
                # Calculate the difference between the two individuals
                difference = individual1 - individual2
                # Calculate the new individual by adding a fraction of the difference to the first individual
                new_individual = individual1 + 0.5 * difference
                # Replace the first individual with the new individual
                self.best_solution = new_individual

# Example usage
def func(x):
    return np.sum(x**2)

swarm_harmony_search = SwarmHarmonySearch(100, 5)
best_solution = swarm_harmony_search(func)
print("Best solution:", best_solution)
print("Fitness value:", func(best_solution))