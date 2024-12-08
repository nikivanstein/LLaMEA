import numpy as np

class APLOS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def adaptive_pareto(self, func, iterations):
        best_func = None
        best_func_value = float('-inf')
        best_dim = None
        best_dim_value = float('-inf')
        for _ in range(iterations):
            func_values = []
            for dim in range(self.dim):
                func_value = func(self.search_space)
                func_values.append(func_value)
            func_values.sort()
            for i, func_value in enumerate(func_values):
                if i == 0:
                    best_func = func
                    best_func_value = func_value
                    best_dim = dim
                    best_dim_value = func_value
                elif func_value > best_func_value:
                    best_func = func
                    best_func_value = func_value
                    best_dim = dim
                    best_dim_value = func_value
            func_value = best_func(best_dim_value)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return best_func, best_func_value, best_dim, best_dim_value

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

aplos = APLOS(1000, 2)  # 1000 function evaluations, 2 dimensions
print(aplos(test_function))  # prints a random value between -10 and 10

def test_pareto_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

aplos_pareto, aplos_pareto_value, aplos_pareto_best_dim, aplos_pareto_best_dim_value = aplos.adaptive_pareto(test_function, 10)
print(f"Best Pareto Function: {aplos_pareto}")
print(f"Best Pareto Value: {aplos_pareto_value}")
print(f"Best Pareto Dimension: {aplos_pareto_best_dim}")
print(f"Best Pareto Dimension Value: {aplos_pareto_best_dim_value}")