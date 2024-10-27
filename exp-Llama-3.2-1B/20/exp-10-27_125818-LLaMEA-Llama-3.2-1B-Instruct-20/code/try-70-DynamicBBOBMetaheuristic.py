import numpy as np

class DynamicBBOBMetaheuristic:
    def __init__(self, budget, dim):
        """
        Initialize the DynamicBBOBMetaheuristic with a given budget and dimensionality.

        Args:
        - budget: The maximum number of function evaluations allowed.
        - dim: The dimensionality of the optimization problem.
        """
        self.budget = budget
        self.dim = dim
        self.population = None
        self.fitness_history = None
        self.logger = None
        self.best_individual = None
        self.best_fitness = None

    def __call__(self, func):
        """
        Optimize the black box function `func` using `self.budget` function evaluations.

        Args:
        - func: The black box function to be optimized.

        Returns:
        - The optimized function value.
        """
        if self.population is None:
            self.population = self.initialize_population(func, self.budget, self.dim)
        else:
            while self.budget > 0:
                # Sample a new individual from the population
                new_individual = self.population[np.random.choice(len(self.population)), :]
                # Evaluate the new individual
                new_fitness = self.evaluate_fitness(new_individual, func)
                # Update the best individual and fitness if necessary
                if new_fitness < self.best_fitness or (new_fitness == self.best_fitness and np.random.rand() < 0.2):
                    self.best_individual = new_individual
                    self.best_fitness = new_fitness
                # Update the population with the new individual
                self.population = np.concatenate((self.population, [new_individual]), axis=0)
                # Update the fitness history
                self.fitness_history = np.append(self.fitness_history, new_fitness)
                # Check if the population has reached the budget
                if len(self.population) >= self.budget:
                    break
                # Sample a new individual from the population
                new_individual = self.population[np.random.choice(len(self.population)), :]
                # Evaluate the new individual
                new_fitness = self.evaluate_fitness(new_individual, func)
                # Update the best individual and fitness if necessary
                if new_fitness < self.best_fitness or (new_fitness == self.best_fitness and np.random.rand() < 0.2):
                    self.best_individual = new_individual
                    self.best_fitness = new_fitness
                # Update the population with the new individual
                self.population = np.concatenate((self.population, [new_individual]), axis=0)
                # Update the fitness history
                self.fitness_history = np.append(self.fitness_history, new_fitness)
                # Check if the population has reached the budget
                if len(self.population) >= self.budget:
                    break
        # Return the optimized function value
        return self.best_fitness

    def initialize_population(self, func, budget, dim):
        """
        Initialize the population with random individuals.

        Args:
        - func: The black box function to be optimized.
        - budget: The maximum number of function evaluations allowed.
        - dim: The dimensionality of the optimization problem.

        Returns:
        - The initialized population.
        """
        return [np.random.uniform(-5.0, 5.0, (dim,)) for _ in range(budget)]

    def evaluate_fitness(self, individual, func):
        """
        Evaluate the fitness of an individual.

        Args:
        - individual: The individual to be evaluated.
        - func: The black box function to be used.

        Returns:
        - The fitness of the individual.
        """
        return func(individual)

# Description: Dynamic BBOB Metaheuristic: An adaptive optimization algorithm that dynamically adjusts its strategy based on the performance of the current population.
# Code: 
# ```python
# import numpy as np
# import random
# from scipy.optimize import minimize
#
# def dynamic_bboo_metaheuristic(func, budget, dim):
#     return DynamicBBOBMetaheuristic(budget, dim)(func)
#
# def func(x):
#     return x[0]**2 + x[1]**2
#
# budget = 1000
# dim = 2
# metaheuristic = dynamic_bboo_metaheuristic(func, budget, dim)
# x0 = [1.0, 1.0]
# res = minimize(func, x0, method='SLSQP', bounds=[(-5.0, 5.0), (-5.0, 5.0)])
# print(f'Optimized function: {res.fun}')
# print(f'Optimized parameters: {res.x}')