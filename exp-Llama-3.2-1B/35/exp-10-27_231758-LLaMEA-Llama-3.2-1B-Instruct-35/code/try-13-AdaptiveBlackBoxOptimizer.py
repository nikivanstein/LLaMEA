import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_univariate

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

        # Refine the strategy using adaptive sampling and line search
        if self.func_evals < self.budget:
            # Use a linear line search to find the optimal point
            def line_search(func, x0, x1, tol=1e-6):
                return minimize(func, x0, method="BFGS", bounds=[(-5.0, 5.0)], tol=tol)

            # Find the optimal point using the line search
            x0 = line_search(func, np.zeros(self.dim), np.zeros(self.dim))
            x1 = line_search(func, np.zeros(self.dim), x0)
            self.func_values = np.concatenate((self.func_values, x0), axis=0)
            self.func_values = np.concatenate((self.func_values, x1), axis=0)

    def evaluate(self, func, x):
        if self.func_evals > 0:
            idx = np.argmin(np.abs(self.func_values))
            return func(self.func_values[idx])
        else:
            return func(x)

# Example usage:
if __name__ == "__main__":
    optimizer = AdaptiveBlackBoxOptimizer(budget=100, dim=10)
    func = lambda x: np.sin(x)
    x0 = np.array([1.0])
    x1 = np.array([2.0])
    result = optimizer(x0, x1)
    print("Optimal point:", result)
    print("Optimal value:", optimizer.evaluate(func, x1))