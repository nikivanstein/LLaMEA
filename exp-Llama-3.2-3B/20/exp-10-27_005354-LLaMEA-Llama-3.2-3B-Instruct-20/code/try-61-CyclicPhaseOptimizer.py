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
        self.refine_probability = 0.2

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

            # Refine the strategy with probability 0.2
            if random.random() < self.refine_probability:
                # Randomly select two individuals to swap
                idx1 = random.randint(0, i)
                idx2 = random.randint(0, i)
                x1, x2 = self.x[idx1], self.x[idx2]
                self.x[idx1] = x2
                self.x[idx2] = x1

        # Return the optimized value
        return self.f_values[-1]

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = CyclicPhaseOptimizer(budget=100, dim=2)
result = optimizer(func)
print(result)