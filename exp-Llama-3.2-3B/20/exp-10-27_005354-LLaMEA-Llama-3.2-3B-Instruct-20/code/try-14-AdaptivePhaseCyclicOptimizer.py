import numpy as np

class AdaptivePhaseCyclicOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the Adaptive-Phase-Cyclic optimizer.

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
        self.step_size = 0.1
        self.line_search = True

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

            # Perform line search
            if self.line_search:
                alpha = self.step_size
                while True:
                    f_value = func(x + alpha * self.shift)
                    if f_value < f_value:
                        self.x[i] = x + alpha * self.shift
                        self.f_values[i] = f_value
                        self.line_search = False
                        break
                    alpha *= 0.9

        # Return the optimized value
        return self.f_values[-1]

# Refine the strategy by changing individual lines with a probability of 0.2
def refine_strategy(algorithm, budget, dim):
    for i in range(budget):
        if np.random.rand() < 0.2:
            # Change the initial point
            algorithm.x[i] = np.random.uniform(-5.0, 5.0, dim)

            # Change the phase and shift
            algorithm.phase = (algorithm.phase + 1) % 3
            if algorithm.phase == 0:
                algorithm.shift = 0
            elif algorithm.phase == 1:
                algorithm.shift = np.random.uniform(0, 1)
            else:
                algorithm.shift = -np.random.uniform(0, 1)

            # Change the current point
            algorithm.x[i] = algorithm.x[i] + algorithm.shift

            # Change the line search
            algorithm.line_search = np.random.rand() < 0.2
            if algorithm.line_search:
                alpha = algorithm.step_size
                while True:
                    f_value = algorithm.func(algorithm.x[i] + alpha * algorithm.shift)
                    if f_value < algorithm.f_values[i]:
                        algorithm.x[i] = algorithm.x[i] + alpha * algorithm.shift
                        algorithm.f_values[i] = f_value
                        algorithm.line_search = False
                        break
                    alpha *= 0.9