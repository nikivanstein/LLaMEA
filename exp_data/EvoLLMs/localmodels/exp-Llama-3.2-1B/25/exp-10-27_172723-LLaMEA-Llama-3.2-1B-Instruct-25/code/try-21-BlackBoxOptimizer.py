import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.iterations = 0

    def __call__(self, func, mutation_prob=0.5):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"

def mutation(individual, mutation_prob):
    # Randomly mutate the individual with probability mutation_prob
    mutated_individual = individual.copy()
    for _ in range(random.randint(0, self.dim)):
        mutated_individual[random.randint(0, self.dim-1)] += random.uniform(-1, 1)
    mutated_individual[random.randint(0, self.dim-1)] *= mutation_prob
    return mutated_individual

# Code:
# ```python
# Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation
# Description: Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation
# Code:
# ```python
# import random
# import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.iterations = 0

    def __call__(self, func, mutation_prob=0.5):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

# Define the function BBOB
def bboB(func, bounds, budget, mutation_prob=0.5):
    # Initialize the Black Box Optimizer
    optimizer = BlackBoxOptimizer(budget, len(bounds))

    # Initialize the population with random solutions
    population = [random.uniform(bounds) for _ in range(100)]

    # Run the optimization algorithm
    for _ in range(100):
        # Evaluate the function for each individual in the population
        fitness_values = [func(individual) for individual in population]

        # Select the fittest individuals
        fittest_individuals = [individual for individual, fitness in zip(population, fitness_values) if fitness > 0.5]

        # Generate new individuals by mutation
        new_individuals = [mutation(individual, mutation_prob) for individual in fittest_individuals]

        # Evaluate the new individuals
        new_fitness_values = [func(individual) for individual in new_individuals]

        # Select the new individuals
        new_individuals = [individual for individual, fitness in zip(new_individuals, new_fitness_values) if fitness > 0.5]

        # Replace the old population with the new population
        population = new_individuals

        # Update the fitness values
        fitness_values = new_fitness_values

        # Check if the budget is exceeded
        if len(population) > budget:
            break

    # Return the fittest individual
    return population[0]

# Define the function to be optimized
def sphere(x):
    return x[0]**2 + x[1]**2

# Evaluate the function
func = sphere
bounds = [-5.0, 5.0]
budget = 100
mutation_prob = 0.5

# Run the optimization algorithm
optimal_solution = bboB(func, bounds, budget, mutation_prob)

# Print the result
print("Optimal solution:", optimal_solution)