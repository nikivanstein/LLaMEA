import numpy as np

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
        self.mutation_prob = 0.2

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

            # Apply mutation with probability 0.2
            if np.random.rand() < self.mutation_prob:
                # Randomly select two points
                j = np.random.randint(0, i)
                k = np.random.randint(j + 1, i + 1)

                # Swap the points
                self.x[j], self.x[k] = self.x[k], self.x[j]

                # Swap the function values
                self.f_values[j], self.f_values[k] = self.f_values[k], self.f_values[j]

        # Return the optimized value
        return self.f_values[-1]