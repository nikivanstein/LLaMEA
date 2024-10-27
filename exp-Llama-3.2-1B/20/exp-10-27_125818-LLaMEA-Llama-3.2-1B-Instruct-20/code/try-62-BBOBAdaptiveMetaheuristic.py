import numpy as np

class BBOBAdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        """
        Initialize the BBOBAdaptiveMetaheuristic with a given budget and dimensionality.

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
        self.population = None
        self.logger = None
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1
        self.population_size = 50

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
                # Initialize the population
                self.population = [self.evaluate_individual() for _ in range(self.population_size)]

                # Evaluate the fitness of each individual
                fitnesses = [self.evaluate_fitness(individual) for individual in self.population]

                # Select the fittest individuals
                self.population = self.select_fittest(population=fitnesses, k=self.population_size)

                # Evaluate the fitness of each individual
                fitnesses = [self.evaluate_fitness(individual) for individual in self.population]

                # Calculate the fitness of the current population
                self.fitnesses = fitnesses

                # Calculate the new population
                self.population = self.create_new_population(population=fitnesses, k=self.population_size)

                # Evaluate the fitness of each individual
                fitnesses = [self.evaluate_fitness(individual) for individual in self.population]

                # Calculate the fitness of the new population
                self.fitnesses = fitnesses

                # Calculate the average fitness
                self.average_fitness = np.mean(self.fitnesses)

                # Update the population
                self.population = self.update_population(population=self.population, fitnesses=self.fitnesses, average_fitness=self.average_fitness)

                # Check if the population has reached the budget
                if self.budget <= 0:
                    break

                # Update the current population
                self.population = self.population[:]

                # Calculate the new fitness of the current individual
                new_fitness = self.func(self.x)

                # Update the individual
                self.x = self.x
                self.f = new_fitness

                # Check if the new fitness is better than the current fitness
                if new_fitness < self.f + 1e-6:  # add a small value to avoid division by zero
                    # Update the individual
                    self.x = self.x
                    self.f = new_fitness
            # Return the optimized function value
            return self.f

# Description: Adaptive Black Box Optimization using Particle Swarm Optimization
# Code: 
# ```python
# import numpy as np
# import random
# from scipy.optimize import minimize
#
# def bboo_adaptive_metaheuristic(func, budget, dim):
#     return BBOBAdaptiveMetaheuristic(budget, dim)(func)
#
# def func(x):
#     return x[0]**2 + x[1]**2
#
# budget = 1000
# dim = 2
# metaheuristic = bboo_adaptive_metaheuristic(func, budget, dim)
# x0 = [1.0, 1.0]
# res = minimize(func, x0, method='SLSQP', bounds=[(-5.0, 5.0), (-5.0, 5.0)])
# print(f'Optimized function: {res.fun}')
# print(f'Optimized parameters: {res.x}')