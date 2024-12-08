import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        # Update strategy based on AOCC score
        if self.func_evals / self.budget < 0.35:
            # Increase exploration by sampling from the entire search space
            idx = np.random.randint(0, self.dim)
            self.func_values[idx] = func(self.func_values[idx])
        else:
            # Decrease exploration by using the adaptive strategy
            idx = np.argmin(np.abs(self.func_values))
            self.func_values[idx] = func(self.func_values[idx])

        # Limit the number of function evaluations
        if self.func_evals > 0:
            self.func_evals -= 1
            if self.func_evals == 0:
                break

# Test the algorithm
optimizer = AdaptiveBlackBoxOptimizer(budget=100, dim=5)
optimizer(func=lambda x: x**2)

# Evaluate the function
print(optimizer(func=lambda x: x**2))  # Should print a value between 0 and 25