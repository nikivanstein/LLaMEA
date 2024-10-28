import numpy as np
import random
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        # Generate random initial population
        return [(np.random.uniform(-5.0, 5.0), np.random.uniform(-5.0, 5.0)) for _ in range(self.population_size)]

    def __call__(self, func):
        # Evaluate the function a specified number of times
        num_evaluations = min(self.budget, self.population_size)
        evaluations = np.random.choice([True, False], num_evaluations, p=[0.5, 0.5])

        # Select the best function based on the evaluations
        best_func = None
        best_score = -np.inf
        for func_eval, evaluation in zip(func, evaluations):
            score = func_eval(func_eval)
            if score > best_score:
                best_func = func_eval
                best_score = score

        # Select a random function from the remaining options
        if best_func is None:
            idx = np.random.randint(0, self.population_size)
            best_func = self.population[idx][0]
            best_score = func(best_func)

        # Optimize the selected function
        bounds = [(self.dim * (-5.0), self.dim * (5.0)) if i == idx else (self.dim * (-5.0), self.dim * (5.0)) for i, (func, _) in enumerate(zip(self.population, func))]
        result = differential_evolution(lambda x: -x[0], bounds, args=(best_func,), maxiter=100, tol=1e-6)
        if result.success:
            # Optimize the function using the selected individual
            func_optimized = best_func
            for _ in range(self.population_size):
                func_eval, _ = differential_evolution(lambda x: -x[0], bounds, args=(func_optimized,), maxiter=100, tol=1e-6)
                if func_eval.x[0] < 0:
                    func_optimized = best_func
                    break

            # Update the population with the optimized function
            self.population = [(func_eval.x[0], func_eval.x[1]) for func_eval, _ in zip(self.population, func_optimized)]
            self.population = self.population[:self.population_size]
        else:
            # If the optimization fails, keep the original population
            self.population = self.population[:self.population_size]

        return func_optimized

# Example usage
budget = 1000
dim = 2
optimizer = BlackBoxOptimizer(budget, dim)
func = lambda x: np.sin(x[0]) + 2 * np.sin(x[1])
best_func = optimizer(func)
print(best_func)