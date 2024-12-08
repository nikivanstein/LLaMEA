import numpy as np
import matplotlib.pyplot as plt

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = []

    def __call__(self, func):
        def func_wrapper(x):
            return func(x)
        self.func_evaluations.append(func_wrapper)
        return func_wrapper

    def optimize(self, func):
        # Initialize the search space
        x = self.search_space[np.random.choice(self.search_space.shape[0], self.budget)]

        # Perform the first evaluation
        func_value = func(x)
        print(f"Initial evaluation: {func_value}")

        # Initialize the best solution and best function value
        best_x = x
        best_func_value = func_value

        # Perform the metaheuristic search
        for _ in range(self.budget):
            # Evaluate the function at the current search point
            func_value = func(x)

            # If the function value is better than the best found so far, update the best solution
            if func_value > best_func_value:
                best_x = x
                best_func_value = func_value

            # If the search space is exhausted, stop the algorithm
            if np.all(x >= self.search_space):
                break

            # Randomly perturb the search point
            x = x + np.random.uniform(-1, 1, self.dim)

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Define a fitness function to evaluate the best solution
def fitness(individual):
    return np.mean(np.sin(individual) + 0.1 * np.cos(2 * individual) + 0.2 * np.sin(3 * individual))

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBO()

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Define a mutation strategy to refine the solution
def mutate(individual):
    if np.random.rand() < 0.4:
        index = np.random.choice(individual.shape[0])
        individual[index] = np.random.uniform(-5.0, 5.0)
    return individual

# Define a selection strategy to select the best individual
def select_best(individuals):
    return individuals[np.argmax([fitness(individual) for individual in individuals])]

# Create a population of 20 individuals
population = [select_best(mutate(individual)) for individual in range(20)]

# Run the genetic algorithm for 1000 generations
for _ in range(1000):
    # Select the best individual
    best_individual = select_best(population)

    # Perform mutation
    best_individual = mutate(best_individual)

    # Evaluate the fitness of the best individual
    fitness_value = fitness(best_individual)

    # Print the best individual and its fitness
    print(f"Best individual: {best_individual}, Fitness: {fitness_value}")

    # Add the best individual to the population
    population.append(best_individual)