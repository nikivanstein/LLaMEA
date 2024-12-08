import numpy as np
import random

class MetaGradientDescent:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-gradient descent algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0

    def __call__(self, func):
        """
        Optimize the black box function `func` using meta-gradient descent.

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

        # Refine the individual lines of the selected solution to refine its strategy
        self.param_values = self.refine_individual_lines(self.param_values)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def refine_individual_lines(self, individual):
        """
        Refine the individual lines of the selected solution to refine its strategy.

        Args:
            individual (np.ndarray): The individual line to refine.

        Returns:
            np.ndarray: The refined individual line.
        """
        # Calculate the average fitness of the individual
        avg_fitness = np.mean(individual)

        # Calculate the standard deviation of the individual
        std_dev = np.std(individual)

        # If the standard deviation is high, mutate the individual
        if std_dev > 3:
            # Generate a new mutation point
            mutation_point = random.randint(0, self.dim - 1)

            # Mutate the individual at the mutation point
            individual[mutation_point] = random.uniform(-5.0, 5.0)

            # Refine the individual lines
            individual = self.refine_individual_lines(individual)

        # Return the refined individual line
        return individual

# Test the MetaGradientDescent algorithm
def test_meta_gradient_descent():
    # Create a new MetaGradientDescent algorithm
    meta_gradient_descent = MetaGradientDescent(budget=100, dim=10, noise_level=0.1)

    # Optimize the black box function
    individual = np.array([-1, -1, -1])
    optimized_individual, func_value = meta_gradient_descent(individual)

    # Print the optimized individual and the objective function value
    print("Optimized Individual:", optimized_individual)
    print("Objective Function Value:", func_value)

# Run the test
test_meta_gradient_descent()