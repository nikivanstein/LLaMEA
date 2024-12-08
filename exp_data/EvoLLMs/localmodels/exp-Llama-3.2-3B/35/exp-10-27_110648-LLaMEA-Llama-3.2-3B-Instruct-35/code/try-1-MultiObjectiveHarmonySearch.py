import numpy as np
import random

class MultiObjectiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 50
        self.harmony_size = 20
        self.pheromone_matrix = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.best_solutions = []
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
            self.best_solutions = population[indices[:self.harmony_size]]

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

            # Refine the solutions using probabilistic refinement
            for i in range(self.population_size):
                if random.random() < 0.35:
                    # Select a random solution from the current population
                    random_solution = population[np.random.choice(self.population_size, 1, replace=False)]
                    # Calculate the fitness difference between the current solution and the random solution
                    fitness_diff = self.fitness_values[i] - self.fitness_values[np.random.choice(self.population_size, 1, replace=False)]
                    # Update the current solution using the fitness difference
                    population[i] = population[i] + fitness_diff * random_solution

            # Check if the best solutions have improved
            for i in range(len(self.best_solutions)):
                for j in range(i+1, len(self.best_solutions)):
                    if self.fitness_values[indices[np.argmin(self.fitness_values[indices])]] > self.fitness_values[indices[np.argmin(self.fitness_values[indices])]]:
                        self.best_solutions[i], self.best_solutions[j] = self.best_solutions[j], self.best_solutions[i]

# Example usage
def func(x):
    return np.sum(x**2)

multi_objective_harmony_search = MultiObjectiveHarmonySearch(100, 5)
best_solutions = multi_objective_harmony_search(func)
print("Best solutions:", best_solutions)
print("Fitness values:", func(best_solutions))