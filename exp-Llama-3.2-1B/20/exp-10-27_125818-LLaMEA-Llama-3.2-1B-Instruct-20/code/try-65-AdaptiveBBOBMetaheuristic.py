# Description: Adaptive Black Box Optimization using Evolutionary Strategies
# Code: 
# ```python
import numpy as np
import random

class AdaptiveBBOBMetaheuristic:
    def __init__(self, budget, dim):
        """
        Initialize the AdaptiveBBOBMetaheuristic with a given budget and dimensionality.

        Args:
        - budget: The maximum number of function evaluations allowed.
        - dim: The dimensionality of the optimization problem.
        """
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = []
        self.fitness_scores = []
        self.logger = None

    def __call__(self, func):
        """
        Optimize the black box function `func` using `self.budget` function evaluations.

        Args:
        - func: The black box function to be optimized.

        Returns:
        - The optimized function value.
        """
        if self.logger is None:
            self.logger = random.getrandbits(1)  # Initialize logger with a random value
        for _ in range(self.budget):
            # Sample a new individual from the population
            individual = random.choice(self.population)
            # Evaluate the function at the new individual
            fitness = self.evaluate_fitness(individual)
            # Update the individual's fitness score
            self.fitness_scores.append(fitness)
            # Update the individual's strategy based on its fitness score
            if self.logger & 1:
                if fitness > 0.2:
                    # Increase the population size
                    self.population.append(individual)
                    # Randomly select a new individual from the population
                    individual = random.choice(self.population)
                else:
                    # Decrease the population size
                    self.population.pop()
                    # Randomly select a new individual from the population
                    individual = random.choice(self.population)
            # Update the individual's fitness value
            individual.fitness = fitness
        # Return the optimized function value
        return individual.fitness

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual.

        Args:
        - individual: The individual to be evaluated.

        Returns:
        - The fitness value of the individual.
        """
        # Calculate the fitness value based on the individual's fitness score
        return individual.fitness

# Description: Adaptive Black Box Optimization using Evolutionary Strategies
# Code: 
# ```python
# import numpy as np
# import random
# from scipy.optimize import minimize
#
# def bboo_metaheuristic(func, budget, dim):
#     return AdaptiveBBOBMetaheuristic(budget, dim)(func)
#
# def func(x):
#     return x[0]**2 + x[1]**2
#
# budget = 1000
# dim = 2
# metaheuristic = bboo_metaheuristic(func, budget, dim)
# x0 = [1.0, 1.0]
# res = minimize(func, x0, method='SLSQP', bounds=[(-5.0, 5.0), (-5.0, 5.0)])
# print(f'Optimized function: {res.fun}')
# print(f'Optimized parameters: {res.x}')