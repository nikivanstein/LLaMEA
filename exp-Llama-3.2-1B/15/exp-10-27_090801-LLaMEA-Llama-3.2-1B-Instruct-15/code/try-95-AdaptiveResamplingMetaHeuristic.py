import numpy as np
import random

class AdaptiveResamplingMetaHeuristic:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the adaptive resampling meta-heuristic algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0
        self.best_individual = None
        self.best_fitness = float('-inf')

    def __call__(self, func):
        """
        Optimize the black box function `func` using adaptive resampling meta-heuristic.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the parameter values to random values within the search space
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)

            # Update the best individual and fitness if the current solution is better
            if func_value > self.best_fitness:
                self.best_individual = self.param_values
                self.best_fitness = func_value

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def select_new_individual(self):
        """
        Select a new individual by resampling from the current population.

        Returns:
            tuple: A tuple containing the selected individual and its fitness.
        """
        # Select a random individual from the current population
        new_individual = np.random.choice(self.param_values, self.dim, replace=False)

        # Update the best individual and fitness if the new individual is better
        if func(new_individual, self.best_individual) > self.best_fitness:
            self.best_individual = new_individual
            self.best_fitness = func(new_individual, self.best_individual)

        # Update the noise level based on the probability of the new individual being better
        self.noise = 0.5 * (1 - self.noise)

        return new_individual, func(new_individual, self.best_individual)

    def mutate(self, individual):
        """
        Mutate the individual by adding or removing a random value within the search space.

        Args:
            individual (numpy.ndarray): The individual to mutate.

        Returns:
            numpy.ndarray: The mutated individual.
        """
        # Select a random value to mutate
        value_to_mutate = random.randint(-5.0, 5.0)

        # Mutate the individual by adding or removing the value
        if random.random() < self.noise:
            individual[value_to_mutate] += np.random.normal(0, 1)

        return individual