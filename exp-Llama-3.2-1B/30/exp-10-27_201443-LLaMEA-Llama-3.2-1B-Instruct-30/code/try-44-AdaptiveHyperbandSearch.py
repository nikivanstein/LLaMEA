import numpy as np

class AdaptiveHyperbandSearch:
    def __init__(self, budget, dim, initial_hyperbands):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.initial_hyperbands = initial_hyperbands
        self.current_hyperband = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Select the current hyperband based on the performance of the individual lines
            if self.current_hyperband < len(self.initial_hyperbands):
                current_hyperband = self.initial_hyperbands[self.current_hyperband]
            else:
                current_hyperband = self.initial_hyperbands[-1]

            # Search within the current hyperband
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break

            # Update the current hyperband if necessary
            if self.func_evaluations / self.budget > current_hyperband / self.budget:
                current_hyperband += 1
                if current_hyperband >= len(self.initial_hyperbands):
                    current_hyperband = 0

            self.func_evaluations += 1
        return func_value

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

adaptive_hyperband_search = AdaptiveHyperbandSearch(1000, 2, [5, 10, 20])
print(adaptive_hyperband_search(test_function))  # prints a random value between -10 and 10