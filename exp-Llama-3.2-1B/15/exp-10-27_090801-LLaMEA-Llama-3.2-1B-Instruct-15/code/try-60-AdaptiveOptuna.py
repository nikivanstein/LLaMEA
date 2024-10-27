# ```python
import numpy as np
import random
import optuna

class AdaptiveOptuna:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the adaptive optuna algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0
        self.best_solution = None
        self.best_score = -np.inf

    def __call__(self, func, iterations=100, memory_limit=100):
        """
        Optimize the black box function `func` using adaptive optuna.

        Args:
            func (callable): The black box function to optimize.
            iterations (int, optional): The number of function evaluations. Defaults to 100.
            memory_limit (int, optional): The maximum number of function evaluations in memory. Defaults to 100.

        Returns:
            tuple: A tuple containing the optimized parameter values, the objective function value, and the number of iterations.
        """
        # Initialize the parameter values to random values within the search space
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)

        # Accumulate noise in the objective function evaluations
        for _ in range(min(self.budget, iterations)):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)

            # Check if the number of iterations is exceeded
            if _ >= iterations:
                break

        # Return the optimized parameter values, the objective function value, and the number of iterations
        return self.param_values, func_value, _ + 1

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

optuna_param = AdaptiveOptuna(budget=1000, dim=2)
optuna_result = optuna.trial.Trial()
optuna_result.start()

optuna_param.run(optuna_result, memory_limit=100)

best_solution = optuna_param.best_solution
best_score = optuna_param.best_score

# Print the results
print(f"Best solution: {best_solution}")
print(f"Best score: {best_score}")

# Save the results to a file
optuna_param.save_to_file(f"optuna_results/{optuna_param.name}_best_solution.npy", f"optuna_results/{optuna_param.name}_best_score.npy")