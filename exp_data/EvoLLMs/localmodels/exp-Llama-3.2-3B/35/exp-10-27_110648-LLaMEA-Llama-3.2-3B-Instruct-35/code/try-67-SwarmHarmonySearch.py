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

            # Refine the best solution with adaptive probability
            if self.fitness_values[indices[0]] < self.fitness_values[indices[1]]:
                if random.random() < self.refinement_probability:
                    # Refine the solution by changing individual lines
                    refined_solution = self.refine_solution(population[indices[0]], self.bounds, self.dim)
                    # Evaluate the fitness of the refined solution
                    refined_fitness = func(refined_solution)
                    # Update the best solution
                    if refined_fitness < self.fitness_values[indices[0]]:
                        self.best_solution = refined_solution

    def refine_solution(self, solution, bounds, dim):
        # Split the solution into individual lines
        lines = solution.reshape(-1, dim)
        # Refine each line with probability 0.35
        refined_lines = []
        for line in lines:
            refined_line = line.copy()
            for i in range(dim):
                if random.random() < self.refinement_probability:
                    # Change the individual line
                    refined_line[i] = np.random.uniform(bounds[0], bounds[1])
            refined_lines.append(refined_line)
        # Reshape the refined solution
        refined_solution = np.array(refined_lines).reshape(-1)
        return refined_solution

# Example usage
def func(x):
    return np.sum(x**2)

swarm_harmony_search = SwarmHarmonySearch(100, 5)
best_solution = swarm_harmony_search(func)
print("Best solution:", best_solution)
print("Fitness value:", func(best_solution))