import numpy as np

class CyclicPhaseRefineOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the Cyclic-Phase refine optimizer.

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
        self.line_search = True
        self.search_prob = 0.2

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

            # Apply line search if probability is met
            if np.random.rand() < self.search_prob:
                # Generate a new point using the current point and shift
                new_x = x + self.shift

                # Evaluate the function at the new point
                new_f_value = func(new_x)

                # Check if the new point is better
                if new_f_value < f_value:
                    # Update the current point and result
                    x = new_x
                    f_value = new_f_value

            # Update the result
            if f_value < self.f_values[i]:
                self.x[i] = x
                self.f_values[i] = f_value

        # Return the optimized value
        return self.f_values[-1]