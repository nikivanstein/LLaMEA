import numpy as np
from scipy.optimize import differential_evolution
import random

class DEBOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the DEBOptimizer with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.func = None

    def __call__(self, func):
        """
        Optimize a black box function using DEBOptimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized function and its value.
        """
        # Get the bounds of the search space
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize the population size and the number of generations
        population_size = 100
        num_generations = 100

        # Initialize the population with random solutions
        self.population = [np.random.uniform(lower_bound, upper_bound, size=(population_size, self.dim)) for _ in range(population_size)]

        # Evaluate the objective function for each individual in the population
        results = []
        for _ in range(num_generations):
            # Evaluate the objective function for each individual in the population
            fitness_values = differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)

            # Select the fittest individuals for the next generation
            fittest_individuals = [self.population[i] for i, _ in enumerate(results) if _ == fitness_values.x[0]]

            # Replace the least fit individuals with the fittest ones
            self.population = [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Update the population with the fittest individuals
            self.population += [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Refine the solution using the selected fittest individuals
            new_individuals = [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]
            new_individuals = random.sample(new_individuals, len(new_individuals) * 0.75)
            new_individuals = [self.evaluate_fitness(individual) for individual in new_individuals]
            self.population = [new_individuals[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Check if the population has reached the budget
            if len(self.population) > self.budget:
                break

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual in the population.

        Args:
            individual (numpy.ndarray): The individual to evaluate.

        Returns:
            float: The fitness value of the individual.
        """
        # Refine the strategy using the selected fittest individuals
        fittest_individuals = [self.population[i] for i, _ in enumerate(self.results) if _ == individual]
        fittest_individual = random.choice(fittest_individuals)
        return -self.func(fittest_individual)

# Initialize the DEBOptimizer with a given budget and dimensionality
optimizer = DEBOptimizer(1000, 10)

# Optimize a black box function using the DEBOptimizer
def optimize_function(func, budget):
    """
    Optimize a black box function using the DEBOptimizer.

    Args:
        func (function): The black box function to optimize.
        budget (int): The maximum number of function evaluations allowed.

    Returns:
        tuple: A tuple containing the optimized function and its value.
    """
    results = []
    for _ in range(budget):
        individual = np.random.uniform(-5.0, 5.0, size=(10,))
        fitness_values = optimizer(func, individual)
        results.append((fitness_values.x[0], -fitness_values.fun))
    return optimizer.func, -optimizer.func

# Optimize a black box function using the DEBOptimizer
def optimize_function_2(func, budget):
    """
    Optimize a black box function using the DEBOptimizer.

    Args:
        func (function): The black box function to optimize.
        budget (int): The maximum number of function evaluations allowed.

    Returns:
        tuple: A tuple containing the optimized function and its value.
    """
    results = []
    for _ in range(budget):
        individual = np.random.uniform(-5.0, 5.0, size=(10,))
        fitness_values = differential_evolution(lambda x: -func(x), individual, bounds=(-5.0, 5.0), x0=individual)
        results.append((fitness_values.x[0], -fitness_values.fun))
    return func, -func

# Test the DEBOptimizer
def test_function(x):
    """
    Test the function x.

    Args:
        x (numpy.ndarray): The input to the function.

    Returns:
        float: The output of the function.
    """
    return x**2 + 2*x + 1

# Optimize the test function using the DEBOptimizer
def optimize_function_3():
    """
    Optimize the test function using the DEBOptimizer.

    Returns:
        tuple: A tuple containing the optimized function and its value.
    """
    return test_function, optimize_function_2(test_function, 1000)

# Test the DEBOptimizer
optimizer, func, _ = optimize_function_3()
print(func(1))
print(func(-1))