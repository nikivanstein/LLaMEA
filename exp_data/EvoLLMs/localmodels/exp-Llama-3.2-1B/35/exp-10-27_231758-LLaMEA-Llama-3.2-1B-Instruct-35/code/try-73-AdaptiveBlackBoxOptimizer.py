import numpy as np

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

    def update_strategy(self, learning_rate, exploration_rate):
        if np.random.rand() < exploration_rate:
            # Exploration strategy: explore the function space
            idx = np.random.choice(self.dim)
            new_func_value = func(self.func_values[idx])
            self.func_values[idx] = new_func_value
            self.func_evals += 1
            if self.func_evals > self.budget:
                break
        else:
            # Exploitation strategy: converge to the best found so far
            idx = np.argmin(np.abs(self.func_values))
            self.func_values[idx] = func(self.func_values[idx])
            self.func_evals -= 1
            if self.func_evals == 0:
                break

        # Refine the strategy using probabilistic learning
        if np.random.rand() < 0.35:
            # Refine the strategy using a small learning rate and high exploration rate
            self.update_strategy(learning_rate=0.01, exploration_rate=0.9)
        else:
            # Refine the strategy using a large learning rate and low exploration rate
            self.update_strategy(learning_rate=0.1, exploration_rate=0.05)

# Initialize the optimizer
optimizer = AdaptiveBlackBoxOptimizer(budget=100, dim=5)

# Define the function to optimize
def func(x):
    return np.sin(x)

# Optimize the function
optimizer(func, func)

# Print the updated strategy
print("Updated strategy:", optimizer.func_values)
print("Updated function value:", optimizer.func_values[-1])