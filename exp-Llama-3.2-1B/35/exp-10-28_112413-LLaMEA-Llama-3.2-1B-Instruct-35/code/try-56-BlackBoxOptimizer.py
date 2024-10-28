import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer.

        Parameters:
        budget (int): The number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize the black box function using the given budget.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        float: The optimized value of the function.
        """
        # Initialize the search space with random values
        x = np.random.uniform(-5.0, 5.0, self.dim)
        
        # Perform the given number of function evaluations
        for _ in range(self.budget):
            # Evaluate the function at the current point
            y = func(x)
            
            # Update the search space if the current function value is better
            if y > x[-1]:
                # Refine the search space using Stochastic Gradient Descent
                x = stgd(x, func, 0.01, 0.1)
        
        # Return the optimized value of the function
        return x[-1]

def stgd(x, func, epsilon, learning_rate):
    """
    Refine the search space using Stochastic Gradient Descent.

    Parameters:
    x (numpy array): The current point in the search space.
    func (function): The black box function to optimize.
    epsilon (float): The step size for the gradient descent update.
    learning_rate (float): The step size for the gradient descent update.

    Returns:
    numpy array: The updated point in the search space.
    """
    y = func(x)
    grad = (y - x[-1]) / epsilon
    x = x[:-1] + np.random.uniform(-epsilon, epsilon, self.dim)
    return x

def func(x):
    return x**2

# One-line description with the main idea
# "Novel Metaheuristic for Solving Black Box Optimization Problems: Iteratively Refining the Search Space using Stochastic Gradient Descent and Evolution Strategies"

# Code
optimizer = BlackBoxOptimizer(1000, 10)
optimized_x = optimizer(func, 0, 1)
print(optimized_x)

# Refine the search space using Evolution Strategies
def evolution_strategy(x, func, population_size, mutation_rate, learning_rate, n_generations):
    """
    Refine the search space using Evolution Strategies.

    Parameters:
    x (numpy array): The current point in the search space.
    func (function): The black box function to optimize.
    population_size (int): The size of the population.
    mutation_rate (float): The probability of mutation.
    learning_rate (float): The step size for the gradient descent update.
    n_generations (int): The number of generations.

    Returns:
    numpy array: The refined point in the search space.
    """
    # Initialize the population with random individuals
    population = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))
    
    for _ in range(n_generations):
        # Evaluate the fitness of each individual
        fitness = [self.__call__(func, individual) for individual in population]
        
        # Select the fittest individuals
        fittest_individuals = np.argsort(fitness)[-population_size:]
        
        # Mutate the fittest individuals
        mutated_individuals = []
        for individual in fittest_individuals:
            mutated_individual = np.copy(individual)
            if random.random() < mutation_rate:
                mutated_individual[random.randint(0, self.dim - 1)] = np.random.uniform(-5.0, 5.0)
            mutated_individuals.append(mutated_individual)
        
        # Replace the least fit individuals with the mutated ones
        population[fittest_individuals] = mutated_individuals
    
    # Return the fittest individual
    return population[np.argmax(fitness)]

# Code
optimized_x = evolution_strategy(optimized_x, func, 100, 0.1, 0.01, 100)
print(optimized_x)