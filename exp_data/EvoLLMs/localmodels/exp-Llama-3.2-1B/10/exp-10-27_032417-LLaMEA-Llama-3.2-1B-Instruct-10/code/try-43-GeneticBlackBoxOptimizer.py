# import numpy as np
# import random

class GeneticBlackBoxOptimizer:
    """
    A metaheuristic algorithm for solving black box optimization problems.
    """

    def __init__(self, budget, dim):
        """
        Initializes the optimizer with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimizes a black box function using the optimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimal solution and the number of function evaluations used.
        """
        # Initialize the population and the number of function evaluations
        population = []
        evaluations = 0

        # Generate an initial population of random solutions
        for _ in range(100):
            population.append(np.random.uniform(-5.0, 5.0, self.dim))

        # Evolve the population over the given number of function evaluations
        while evaluations < self.budget:
            # Evaluate the fitness of each individual in the population
            fitnesses = [func(individual) for individual in population]

            # Select the fittest individuals
            fittest_individuals = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)[:self.budget//2]

            # Create a new population by mutating the fittest individuals
            new_population = []
            for i in range(self.budget//2):
                # Randomly select an individual from the fittest individuals
                individual = fittest_individuals[i]

                # Generate a new individual by mutating the current individual
                mutated_individual = individual.copy()
                if np.random.rand() < 0.1:
                    # Apply adaptive mutation strategy
                    mutated_individual += np.random.uniform(-1.0, 1.0, self.dim)

                # Evaluate the fitness of the new individual
                fitness = func(mutated_individual)

                # Add the new individual to the new population
                new_population.append(mutated_individual)

            # Replace the old population with the new population
            population = new_population

            # Evaluate the fitness of each individual in the new population
            fitnesses = [func(individual) for individual in population]

            # Select the fittest individuals
            fittest_individuals = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)[:self.budget//2]

            # Create a new population by mutating the fittest individuals
            new_population = []
            for i in range(self.budget//2):
                # Randomly select an individual from the fittest individuals
                individual = fittest_individuals[i]

                # Generate a new individual by mutating the current individual
                mutated_individual = individual.copy()
                if np.random.rand() < 0.1:
                    # Apply adaptive mutation strategy
                    mutated_individual += np.random.uniform(-1.0, 1.0, self.dim)

                # Evaluate the fitness of the new individual
                fitness = func(mutated_individual)

                # Add the new individual to the new population
                new_population.append(mutated_individual)

            # Replace the old population with the new population
            population = new_population

        # Return the optimal solution and the number of function evaluations used
        return population[0], evaluations


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = GeneticBlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

# Novel heuristic algorithm: 
# Genetic Algorithm for Black Box Optimization with Adaptive Mutation Strategy
# 
# Description: A novel metaheuristic algorithm combining genetic algorithm and simulated annealing to optimize black box functions.
# 
# Code: 
# ```python
# import numpy as np
# import random

class GeneticBlackBoxOptimizer:
    """
    A metaheuristic algorithm for solving black box optimization problems.
    """

    def __init__(self, budget, dim):
        """
        Initializes the optimizer with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimizes a black box function using the optimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimal solution and the number of function evaluations used.
        """
        # Initialize the population and the number of function evaluations
        population = []
        evaluations = 0

        # Generate an initial population of random solutions
        for _ in range(100):
            population.append(np.random.uniform(-5.0, 5.0, self.dim))

        # Evolve the population over the given number of function evaluations
        while evaluations < self.budget:
            # Evaluate the fitness of each individual in the population
            fitnesses = [func(individual) for individual in population]

            # Select the fittest individuals
            fittest_individuals = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)[:self.budget//2]

            # Create a new population by mutating the fittest individuals
            new_population = []
            for i in range(self.budget//2):
                # Randomly select an individual from the fittest individuals
                individual = fittest_individuals[i]

                # Generate a new individual by mutating the current individual
                mutated_individual = individual.copy()
                if np.random.rand() < 0.1:
                    # Apply adaptive mutation strategy
                    mutated_individual += np.random.uniform(-1.0, 1.0, self.dim)

                # Evaluate the fitness of the new individual
                fitness = func(mutated_individual)

                # Add the new individual to the new population
                new_population.append(mutated_individual)

            # Replace the old population with the new population
            population = new_population

            # Evaluate the fitness of each individual in the new population
            fitnesses = [func(individual) for individual in population]

            # Select the fittest individuals
            fittest_individuals = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)[:self.budget//2]

            # Create a new population by mutating the fittest individuals
            new_population = []
            for i in range(self.budget//2):
                # Randomly select an individual from the fittest individuals
                individual = fittest_individuals[i]

                # Generate a new individual by mutating the current individual
                mutated_individual = individual.copy()
                if np.random.rand() < 0.1:
                    # Apply adaptive mutation strategy
                    mutated_individual += np.random.uniform(-1.0, 1.0, self.dim)

                # Evaluate the fitness of the new individual
                fitness = func(mutated_individual)

                # Add the new individual to the new population
                new_population.append(mutated_individual)

            # Replace the old population with the new population
            population = new_population

        # Return the optimal solution and the number of function evaluations used
        return population[0], evaluations