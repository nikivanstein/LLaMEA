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
        self.proportion = 0.2

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

    def refine_strategy(self):
        """
        Refine the strategy by changing the individual lines of the selected solution.
        """
        # Refine the initial point
        self.x = self.x * (1 + random.uniform(-self.proportion, self.proportion))

        # Refine the phase and shift
        self.phase = (self.phase + random.randint(0, 1)) % 3
        if self.phase == 0:
            self.shift = random.uniform(0, 1)
        elif self.phase == 1:
            self.shift = np.random.uniform(0, 1)
        else:
            self.shift = -np.random.uniform(0, 1)

        # Refine the updated point
        self.x = self.x + self.shift * random.uniform(-self.proportion, self.proportion)

        # Refine the result
        self.f_values = np.minimum(self.f_values, func(self.x))