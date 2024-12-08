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
        self.mutation_rate = 0.2

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

            # Apply mutation
            if random.random() < self.mutation_rate:
                # Randomly select two points
                idx1, idx2 = random.sample(range(self.budget), 2)

                # Calculate the mutation factor
                mu = np.random.uniform(-1, 1)

                # Update the points
                self.x[idx1] += mu * (self.x[idx2] - self.x[idx1])
                self.f_values[idx1] = func(self.x[idx1])

                # Update the best point
                if self.f_values[idx1] < self.f_values[idx2]:
                    self.x[idx2] = self.x[idx1]
                    self.f_values[idx2] = self.f_values[idx1]

            # Evaluate the function at the updated point
            f_value = func(self.x[i])

            # Update the result
            if f_value < self.f_values[i]:
                self.x[i] = self.x[i]
                self.f_values[i] = f_value

        # Return the optimized value
        return self.f_values[-1]