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
        self.step_size = 1.0
        self.learning_rate = 0.01

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

            # Calculate the gradient of the objective function with respect to the current parameter values
            gradient = np.gradient(func_value)

            # Update the parameter values based on the gradient and the current best solution
            self.param_values += self.learning_rate * gradient / self.step_size

            # Update the step size based on the gradient direction
            if np.linalg.norm(gradient) > self.step_size * self.noise_level:
                self.step_size *= 0.9

            # Update the noise level based on the fitness landscape
            if func_value < self.param_values[0] - 2.0 * self.step_size * self.noise_level:
                self.noise += 0.1
            elif func_value > self.param_values[-1] + 2.0 * self.step_size * self.noise_level:
                self.noise -= 0.1

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

# Example usage:
if __name__ == "__main__":
    # Create an instance of the meta-gradient descent algorithm
    mgd = MetaGradientDescent(budget=1000, dim=10, noise_level=0.1)

    # Optimize the function f(x) = x^2 + 2x + 1 using the meta-gradient descent algorithm
    func = lambda x: x**2 + 2*x + 1
    optimized_params, optimized_func_value = mgd(func)

    # Print the optimized parameter values and the objective function value
    print("Optimized parameters:", optimized_params)
    print("Optimized function value:", optimized_func_value)