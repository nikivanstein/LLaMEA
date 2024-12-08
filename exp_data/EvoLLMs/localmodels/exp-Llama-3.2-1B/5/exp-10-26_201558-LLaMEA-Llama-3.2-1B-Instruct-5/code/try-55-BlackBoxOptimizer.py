# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
import numpy as np
import random
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a given budget and dimensionality.

        Parameters:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize the black box function `func` using the given budget and search space.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population size
        population_size = 100

        # Initialize the population with random parameters
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))

        # Evaluate the function for each individual in the population
        for _ in range(self.budget):
            # Evaluate the function for each individual in the population
            func_values = func(population)

            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

        # Define the mutation strategy
        def mutation(individual):
            # Randomly select a dimension
            dim_idx = random.randint(0, self.dim - 1)

            # Randomly flip the value at the selected dimension
            individual[dim_idx] = 1 - individual[dim_idx]

        # Define the selection strategy
        def selection(population):
            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func(population))[:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            return new_population

        # Define the crossover strategy
        def crossover(parent1, parent2):
            # Select a random crossover point
            crossover_point = random.randint(0, self.dim - 1)

            # Create a new offspring by combining the two parents
            offspring = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])

            # Replace the old parents with the new offspring
            return offspring

        # Define the mutation rate
        mutation_rate = 0.05

        # Initialize the best function value and the best individual
        best_func_value = np.inf
        best_individual = None

        # Iterate over the population to find the best individual
        for _ in range(self.budget):
            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func(population))[:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

            # Evaluate the new function value
            func_value = func(new_population)

            # Check if the new function value is better than the best function value
            if func_value < best_func_value:
                # Update the best function value and the best individual
                best_func_value = func_value
                best_individual = new_population

        # Return the optimized parameters and the optimized function value
        return best_individual, best_func_value

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
# BlackBoxOptimizer: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# ```python
# ```python
# import numpy as np
# import random
# import scipy.optimize as optimize
# import copy

class GeneticProgrammingBlackBoxOptimizer(BlackBoxOptimizer):
    def __init__(self, budget, dim):
        """
        Initialize the GeneticProgrammingBlackBoxOptimizer with a given budget and dimensionality.

        Parameters:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        super().__init__(budget, dim)

    def __call__(self, func):
        """
        Optimize the black box function `func` using the given budget and search space.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population size
        population_size = 100

        # Initialize the population with random parameters
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))

        # Evaluate the function for each individual in the population
        for _ in range(self.budget):
            # Evaluate the function for each individual in the population
            func_values = func(population)

            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

        # Define the mutation strategy
        def mutation(individual):
            # Randomly select a dimension
            dim_idx = random.randint(0, self.dim - 1)

            # Randomly flip the value at the selected dimension
            individual[dim_idx] = 1 - individual[dim_idx]

        # Define the selection strategy
        def selection(population):
            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func(population))[:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            return new_population

        # Define the crossover strategy
        def crossover(parent1, parent2):
            # Select a random crossover point
            crossover_point = random.randint(0, self.dim - 1)

            # Create a new offspring by combining the two parents
            offspring = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])

            # Replace the old parents with the new offspring
            return offspring

        # Define the mutation rate
        mutation_rate = 0.05

        # Initialize the best function value and the best individual
        best_func_value = np.inf
        best_individual = None

        # Iterate over the population to find the best individual
        for _ in range(self.budget):
            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func(population))[:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

            # Evaluate the new function value
            func_value = func(new_population)

            # Check if the new function value is better than the best function value
            if func_value < best_func_value:
                # Update the best function value and the best individual
                best_func_value = func_value
                best_individual = new_population

        # Return the optimized parameters and the optimized function value
        return best_individual, best_func_value

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
# GeneticProgrammingBlackBoxOptimizer: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# ```python
# ```python
# import numpy as np
# import random
# import scipy.optimize as optimize
# import copy

def optimize_bbo(func, budget, dim):
    """
    Optimize the black box function `func` using the given budget and search space.

    Parameters:
    func (function): The black box function to optimize.
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.

    Returns:
    tuple: The optimized parameters and the optimized function value.
    """
    # Initialize the population size
    population_size = 100

    # Initialize the population with random parameters
    population = np.random.uniform(-5.0, 5.0, (population_size, dim))

    # Define the mutation strategy
    def mutation(individual):
        # Randomly select a dimension
        dim_idx = random.randint(0, dim - 1)

        # Randomly flip the value at the selected dimension
        individual[dim_idx] = 1 - individual[dim_idx]

    # Define the selection strategy
    def selection(population):
        # Select the fittest individuals based on the function values
        fittest_individuals = np.argsort(func(population))[:population_size // 2]

        # Create a new population by combining the fittest individuals
        new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

        # Replace the old population with the new population
        return new_population

    # Define the crossover strategy
    def crossover(parent1, parent2):
        # Select a random crossover point
        crossover_point = random.randint(0, dim - 1)

        # Create a new offspring by combining the two parents
        offspring = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])

        # Replace the old parents with the new offspring
        return offspring

    # Define the mutation rate
    mutation_rate = 0.05

    # Initialize the best function value and the best individual
    best_func_value = np.inf
    best_individual = None

    # Iterate over the population to find the best individual
    for _ in range(budget):
        # Select the fittest individuals based on the function values
        fittest_individuals = np.argsort(func(population))[:population_size // 2]

        # Create a new population by combining the fittest individuals
        new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

        # Replace the old population with the new population
        population = new_population

        # Evaluate the new function value
        func_value = func(new_population)

        # Check if the new function value is better than the best function value
        if func_value < best_func_value:
            # Update the best function value and the best individual
            best_func_value = func_value
            best_individual = new_population

    # Return the optimized parameters and the optimized function value
    return best_individual, best_func_value

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
# import numpy as np
# import random
# import scipy.optimize as optimize
# import copy

# Evaluate the black box function
def evaluate_bbo(func, x):
    """
    Evaluate the black box function `func` at the given point `x`.

    Parameters:
    func (function): The black box function to evaluate.
    x (array): The point at which to evaluate the function.

    Returns:
    float: The value of the function at the given point.
    """
    return func(x)

# Example usage:
budget = 100
dim = 10
best_individual, best_func_value = optimize_bbo(evaluate_bbo, budget, dim)
print(f"Best individual: {best_individual}")
print(f"Best function value: {best_func_value}")