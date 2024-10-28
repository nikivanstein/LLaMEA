import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1

    def __call__(self, func):
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)
        
        # Replace the old population with the new one
        self.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        return new_population[best_individual]

    def optimize(self, func, budget, dim):
        """
        Optimize the black box function using differential evolution.

        Parameters:
        func (function): The black box function to optimize.
        budget (int): The number of function evaluations.
        dim (int): The dimensionality of the search space.

        Returns:
        individual (float): The best individual in the new population.
        """
        # Initialize the new population
        new_population = self.__call__(func)
        
        # Perform differential evolution
        result = differential_evolution(self.f, [(x - 0.5, x + 0.5) for x in new_population], x0_bounds=[(-5.0, 5.0), (-5.0, 5.0)], bounds=[(-5.0, 5.0), (-5.0, 5.0)], n_iter=100)
        
        # Replace the old population with the new one
        self.population = new_population
        
        # Return the best individual
        return result.x[0]

    def f(self, x):
        """
        The function to optimize.

        Parameters:
        x (float): The individual to optimize.

        Returns:
        float: The fitness of the individual.
        """
        return func(x)

# Description: A novel metaheuristic algorithm for black box optimization.
# Code: 