import numpy as np
import random

class CyclicPhaseOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the Cyclic-Phase optimizer.

        Parameters:
        budget (int): Maximum number of function evaluations.
        dim (int): Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.x = np.zeros((budget, dim))
        self.f_values = np.zeros(budget)
        self.phase = 0
        self.shift = 0

    def __call__(self, func):
        """
        Optimize the given black box function.

        Parameters:
        func (callable): The black box function to optimize.

        Returns:
        float: The optimized value of the function.
        """
        for i in range(self.budget):
            # Generate a random initial point
            x = np.random.uniform(-5.0, 5.0, self.dim)

            # Evaluate the function at the initial point
            f_value = func(x)

            # Store the result
            self.x[i] = x
            self.f_values[i] = f_value

            # Update the phase and shift
            self.phase = (self.phase + 1) % 3
            if self.phase == 0:
                self.shift = 0
            elif self.phase == 1:
                self.shift = np.random.uniform(0, 1)
            else:
                self.shift = -np.random.uniform(0, 1)

            # Update the current point
            x = x + self.shift

            # Evaluate the function at the updated point
            f_value = func(x)

            # Update the result
            if f_value < self.f_values[i]:
                self.x[i] = x
                self.f_values[i] = f_value

        # Return the optimized value
        return self.f_values[-1]

    def refine(self):
        """
        Refine the strategy of the Cyclic-Phase optimizer.

        This method changes 20% of the individuals in the current population.
        """
        new_x = np.copy(self.x)
        new_f_values = np.copy(self.f_values)

        for i in range(self.budget):
            if random.random() < 0.2:
                # Change the individual
                new_x[i] = self.x[i] + np.random.uniform(-1, 1, self.dim)
                new_f_values[i] = func(new_x[i])

        # Update the population
        self.x = new_x
        self.f_values = new_f_values

# Usage
if __name__ == "__main__":
    # Define the black box function
    def func(x):
        return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.sin(10*x[1])

    # Initialize the Cyclic-Phase optimizer
    optimizer = CyclicPhaseOptimizer(100, 2)

    # Optimize the function
    optimized_value = optimizer(func)

    print("Optimized value:", optimized_value)