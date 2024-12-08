import numpy as np

class AMGD:
    def __init__(self, budget, dim, learning_rate, multi_step_size, alpha):
        self.budget = budget
        self.dim = dim
        self.learning_rate = learning_rate
        self.multi_step_size = multi_step_size
        self.alpha = alpha
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Select a multi-step search strategy based on the current function value
            if np.abs(func(self.search_space)) < 1e-6:  # stop if the function value is close to zero
                return func(self.search_space)
            # Perform a single step gradient descent
            step = np.zeros(self.dim)
            for i in range(self.dim):
                step[i] = (func(self.search_space[i]) - func(self.search_space[i-1])) / self.multi_step_size
            # Update the search direction using the adaptive learning rate
            self.learning_rate = max(0.01, min(self.learning_rate, 0.1 * np.linalg.norm(step) / self.alpha))
            # Update the search space using the adaptive multi-step size
            self.search_space = np.linspace(self.search_space[0] - self.multi_step_size, self.search_space[0] + self.multi_step_size, self.dim)
            self.func_evaluations += 1
        return func(self.search_space)