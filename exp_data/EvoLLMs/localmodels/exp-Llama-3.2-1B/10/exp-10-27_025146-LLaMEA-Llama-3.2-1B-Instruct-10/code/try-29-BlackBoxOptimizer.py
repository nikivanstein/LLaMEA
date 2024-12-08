import random
import numpy as np
import copy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func, mutation_rate=0.1, crossover_rate=0.1):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.
            mutation_rate (float, optional): The rate at which individuals are mutated. Defaults to 0.1.
            crossover_rate (float, optional): The rate at which individuals are crossed over. Defaults to 0.1.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Generate a random point in the search space
            point = copy.deepcopy(self.search_space[np.random.randint(0, self.dim)])

            # Evaluate the function at the current point
            value = func(point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = point

            # Apply mutation
            if random.random() < mutation_rate:
                # Randomly select an individual from the population
                individual = copy.deepcopy(self.population[best_index])

                # Randomly select two parents from the population
                parent1 = copy.deepcopy(self.population[np.random.randint(0, len(self.population))])
                parent2 = copy.deepcopy(self.population[np.random.randint(0, len(self.population))])

                # Perform crossover
                if random.random() < crossover_rate:
                    # Randomly select a crossover point
                    crossover_point = np.random.randint(0, len(individual))

                    # Split the individual into two parts
                    child1 = copy.deepcopy(individual[:crossover_point])
                    child2 = copy.deepcopy(individual[crossover_point:])

                    # Combine the two parts
                    child = copy.deepcopy(child1)
                    child[crossover_point:] = parent2[crossover_point:]

                    # Replace the original individual with the new child
                    self.population[best_index] = child

    def generate_population(self, num_individuals):
        """
        Generate a population of random individuals.

        Args:
            num_individuals (int): The number of individuals in the population.

        Returns:
            list: A list of individuals in the population.
        """
        population = []
        for _ in range(num_individuals):
            individual = copy.deepcopy(self.search_space[np.random.randint(0, self.dim)])
            population.append(individual)

        return population

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 