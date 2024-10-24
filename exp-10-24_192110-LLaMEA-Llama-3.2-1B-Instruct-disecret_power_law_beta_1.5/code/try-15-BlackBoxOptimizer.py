import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []

    def __call__(self, func):
        # Initialize population with random solutions
        for _ in range(self.budget):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the current solution
            score = func(solution)
            # Add the solution and its score to the population
            self.population.append((solution, score))

        # Select the best solution based on the budget
        best_solution = self.population[0]
        best_score = best_solution[1]
        for solution, score in self.population:
            if score > best_score:
                best_solution = solution
                best_score = score

        return best_solution

    def evolve(self, num_generations):
        # Evolve the population using mutation and selection
        for _ in range(num_generations):
            # Select the best solution
            best_solution = self.__call__(self.func)

            # Generate a new population by mutation and selection
            new_population = []
            while len(new_population) < self.budget:
                # Randomly select a solution from the current population
                solution = self.population[-1]
                # Generate a new solution by perturbing the current solution
                new_solution = solution + np.random.uniform(-1.0, 1.0, self.dim)
                # Evaluate the new solution
                new_score = self.func(new_solution)
                # Add the new solution and its score to the new population
                new_population.append((new_solution, new_score))

            # Replace the old population with the new population
            self.population = new_population

        # Return the best solution found
        return best_solution

# Define the function to be optimized
def func(x):
    return x**2 + 2*x + 1

# Create an instance of the BlackBoxOptimizer class
optimizer = BlackBoxOptimizer(100, 2)

# Evolve the population for 100 generations
best_solution = optimizer.evolve(100)

# Print the best solution found
print("Best solution:", best_solution)
print("Score:", func(best_solution))