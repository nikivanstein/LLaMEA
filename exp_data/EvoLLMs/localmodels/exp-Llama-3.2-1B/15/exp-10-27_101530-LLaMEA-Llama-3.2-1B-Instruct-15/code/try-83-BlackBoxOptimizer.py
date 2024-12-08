import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.search_space_history = [(self.search_space[0], self.search_space[1])]

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
            # Update the search space history
            self.search_space_history.append((self.search_space[0], self.search_space[1]))

        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

def novelty_search(budget, dim, search_space, func, iterations=1000):
    """
    Novel Metaheuristic Algorithm for Black Box Optimization (NMBAO)
    """
    # Initialize the algorithm
    optimizer = BlackBoxOptimizer(budget, dim)

    # Initialize the population
    population = [optimizer.search_space[0], optimizer.search_space[1]] * iterations

    # Run the algorithm for the specified number of iterations
    for _ in range(iterations):
        # Evaluate the fitness of each individual in the population
        fitness_values = [optimizer(func(individual)) for individual in population]

        # Select the fittest individuals
        fittest_individuals = population[np.argsort(fitness_values)[::-1][:budget]]

        # Create a new population by mutating the fittest individuals
        new_population = [individual for individual in fittest_individuals]
        for _ in range(dim):
            mutation_rate = 0.01
            for i in range(len(new_population)):
                if random.random() < mutation_rate:
                    new_population[i] = (new_population[i] + random.uniform(-1, 1)) / 2

        # Evaluate the fitness of each individual in the new population
        fitness_values = [optimizer.func(individual) for individual in new_population]

        # Replace the old population with the new population
        population = new_population

        # Update the search space history
        for i in range(len(population)):
            for _ in range(10):
                point = (population[i] + random.uniform(-1, 1)) / 2
                if point not in population:
                    population.append(point)

    # Return the best individual found
    return population[0]

# Example usage:
# Define the function to optimize
def func(x):
    return x[0]**2 + x[1]**2

# Run the NMBAO algorithm
best_individual = novelty_search(100, 2, (-5.0, 5.0), func)

# Print the result
print("Best individual:", best_individual)
print("Fitness:", func(best_individual))