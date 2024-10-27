# Description: Adaptive Black Box Optimization using Evolutionary Search
# Code: 
# ```python
import numpy as np
import random
import operator
import time

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        """
        Initialize the BBOBMetaheuristic with a given budget and dimensionality.

        Args:
        - budget: The maximum number of function evaluations allowed.
        - dim: The dimensionality of the optimization problem.
        """
        self.budget = budget
        self.dim = dim
        self.func = None
        self.space = None
        self.x = None
        self.f = None

    def __call__(self, func):
        """
        Optimize the black box function `func` using `self.budget` function evaluations.

        Args:
        - func: The black box function to be optimized.

        Returns:
        - The optimized function value.
        """
        if self.func is None:
            self.func = func
            self.space = np.random.uniform(-5.0, 5.0, (self.dim,))
            self.x = np.random.uniform(-5.0, 5.0, (self.dim,))
            self.f = self.func(self.x)
        else:
            while self.budget > 0:
                # Sample a new point in the search space
                self.x = np.random.uniform(-5.0, 5.0, (self.dim,))
                # Evaluate the function at the new point
                self.f = self.func(self.x)
                # Check if the new point is better than the current point
                if self.f < self.f + 1e-6:  # add a small value to avoid division by zero
                    # Update the current point
                    self.x = self.x
                    self.f = self.f
            # Return the optimized function value
            return self.f

    def evolve(self, new_individual, population_size, mutation_rate, bounds):
        """
        Evolve the population using crossover and mutation.

        Args:
        - new_individual: The new individual to be added to the population.
        - population_size: The size of the population.
        - mutation_rate: The rate of mutation.
        - bounds: The bounds of the search space.

        Returns:
        - The evolved population.
        """
        # Select the fittest individuals
        fittest_individuals = sorted(self.population, key=self.f, reverse=True)[:self.population_size // 2]
        # Create a new population
        new_population = []
        for _ in range(population_size):
            # Randomly select two parents from the fittest individuals
            parent1, parent2 = random.sample(fittest_individuals, 2)
            # Perform crossover
            child = self.crossover(parent1, parent2)
            # Perform mutation
            child = self.mutation(child, mutation_rate, bounds)
            # Add the child to the new population
            new_population.append(child)
        # Return the new population
        return new_population

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.

        Args:
        - parent1: The first parent.
        - parent2: The second parent.

        Returns:
        - The child.
        """
        # Select a random crossover point
        crossover_point = np.random.randint(0, len(parent1))
        # Split the parents at the crossover point
        child1 = parent1[:crossover_point]
        child2 = parent2[crossover_point:]
        # Combine the children
        child = child1 + child2
        # Return the child
        return child

    def mutation(self, individual, mutation_rate, bounds):
        """
        Perform mutation on an individual.

        Args:
        - individual: The individual to be mutated.
        - mutation_rate: The rate of mutation.
        - bounds: The bounds of the search space.

        Returns:
        - The mutated individual.
        """
        # Select a random point to mutate
        mutation_point = np.random.randint(0, len(individual))
        # Flip the bit at the mutation point
        individual[mutation_point] = 1 - individual[mutation_point]
        # Return the mutated individual
        return individual

# Description: Adaptive Black Box Optimization using Evolutionary Search
# Code: 
# ```python
bboo_metaheuristic = BBOBMetaheuristic(1000, 2)
new_individual = [1.0, 1.0]
population_size = 100
mutation_rate = 0.1
bounds = [-5.0, 5.0]
evolved_population = bboo_metaheuristic.evolve(new_individual, population_size, mutation_rate, bounds)
print(f'Optimized function: {bboo_metaheuristic.func evolved_population[0]}')
print(f'Optimized parameters: {bboo_metaheuristic.x evolved_population[0]}')