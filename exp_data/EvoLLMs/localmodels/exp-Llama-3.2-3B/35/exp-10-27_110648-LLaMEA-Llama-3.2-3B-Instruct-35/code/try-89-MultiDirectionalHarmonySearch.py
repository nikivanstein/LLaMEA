import numpy as np
import random

class MultiDirectionalHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 50
        self.harmony_size = 20
        self.pheromone_matrix = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.best_solution = None
        self.fitness_values = np.zeros(self.population_size)
        self.refine_probability = 0.35

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
                new_solution = population[i] + self.bounds[1] - self.bounds[0] * difference
                # Refine the solution with probability 0.35
                if random.random() < self.refine_probability:
                    new_solution = self.refine_solution(new_solution, population, self.bounds)
                population[i] = new_solution

            # Evaluate the fitness of the new solutions
            self.fitness_values = func(population)

            # Update the pheromone matrix
            for i in range(self.population_size):
                self.pheromone_matrix[i] = self.fitness_values[i] / np.sum(self.fitness_values)

            # Check if the best solution has improved
            if self.fitness_values[indices[0]] < self.fitness_values[indices[1]]:
                self.best_solution = population[indices[0]]

    def refine_solution(self, new_solution, population, bounds):
        # Refine the solution by perturbing its lines
        refined_solution = np.copy(new_solution)
        for i in range(self.dim):
            if random.random() < self.refine_probability:
                refined_solution[i] += np.random.uniform(-bounds[1] - bounds[0], bounds[1] - bounds[0])
        return refined_solution

# Example usage
def func(x):
    return np.sum(x**2)

multi_directional_harmony_search = MultiDirectionalHarmonySearch(100, 5)
best_solution = multi_directional_harmony_search(func)
print("Best solution:", best_solution)
print("Fitness value:", func(best_solution))