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

            # Refine the strategy with mutation
            if np.random.rand() < self.mutation_prob:
                # Randomly select a dimension to mutate
                idx = np.random.randint(self.dim)
                # Generate a new value for the dimension
                new_value = np.random.uniform(-5.0, 5.0)
                # Update the dimension
                self.x[i, idx] = new_value

        # Return the optimized value
        return self.f_values[-1]

# Example usage:
def func(x):
    return np.sum(x**2)

optimizer = CyclicPhaseOptimizer(100, 5)
result = optimizer(func)
print(result)