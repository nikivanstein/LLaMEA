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

            # Refine the strategy with probabilistic refinement
            refine_indices = np.random.choice(self.population_size, size=int(self.population_size * 0.35), replace=False)
            refined_population = population.copy()
            for i in refine_indices:
                refined_solution = population[i]
                # Randomly choose a dimension to refine
                dim_to_refine = np.random.randint(0, self.dim)
                # Randomly choose a step size for the dimension
                step_size = np.random.uniform(-1, 1)
                refined_solution[dim_to_refine] += step_size
                # Ensure the solution stays within the bounds
                refined_solution = np.clip(refined_solution, self.bounds[0], self.bounds[1])
                refined_population[i] = refined_solution
            # Evaluate the fitness of the refined solutions
            refined_fitness_values = func(refined_population)
            # Update the pheromone matrix
            for i in range(self.population_size):
                self.pheromone_matrix[i] = refined_fitness_values[i] / np.sum(refined_fitness_values)

            # Check if the best solution has improved
            if refined_fitness_values[indices[0]] < refined_fitness_values[indices[1]]:
                self.best_solution = refined_population[indices[0]]

# Example usage
def func(x):
    return np.sum(x**2)

swarm_harmony_search = SwarmHarmonySearch(100, 5)
best_solution = swarm_harmony_search(func)
print("Best solution:", best_solution)
print("Fitness value:", func(best_solution))