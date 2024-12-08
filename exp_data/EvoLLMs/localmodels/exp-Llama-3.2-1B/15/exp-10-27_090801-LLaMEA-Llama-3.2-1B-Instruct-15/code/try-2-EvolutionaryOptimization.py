import numpy as np
import random

class EvolutionaryOptimization:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the evolutionary optimization algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0
        self.population_size = 100
        self.mutation_rate = 0.1

    def __call__(self, func):
        """
        Optimize the black box function `func` using evolutionary optimization.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population with random parameter values
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.population + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            self.population += self.noise * np.random.normal(0, 1, self.dim)

        # Select the fittest individual
        self.fittest_individual = np.argmax(self.population)

        # Return the optimized parameter values and the objective function value
        return self.population[self.fittest_individual], func_value

    def mutate(self, individual):
        """
        Mutate an individual by changing a random parameter value.

        Args:
            individual (numpy.ndarray): The individual to mutate.

        Returns:
            numpy.ndarray: The mutated individual.
        """
        # Randomly select a parameter value to mutate
        idx = np.random.randint(0, self.dim)

        # Change the parameter value
        individual[idx] += random.uniform(-1, 1)

        return individual

# Example usage:
if __name__ == "__main__":
    # Define the black box function
    def func(x):
        return x[0]**2 + x[1]**2

    # Create an instance of the evolutionary optimization algorithm
    optimization = EvolutionaryOptimization(100, 2)

    # Optimize the function
    optimized_params, optimized_value = optimization(func)

    # Print the results
    print("Optimized Parameters:", optimized_params)
    print("Optimized Value:", optimized_value)