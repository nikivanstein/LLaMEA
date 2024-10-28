import numpy as np

class DABU:
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

    def refine_strategy(self):
        # Refine the search space by increasing the lower bound
        self.search_space[0] = -10.0
        # Refine the search space by decreasing the upper bound
        self.search_space[1] = 5.0

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10

# Update the DABU algorithm with a novel metaheuristic strategy
class DABURefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.refine_strategy()

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        self.refine_strategy()
        return func_value

    def refine_strategy(self):
        # Refine the search space by increasing the lower bound
        self.search_space[0] = -10.0
        # Refine the search space by decreasing the upper bound
        self.search_space[1] = 5.0

# Evaluate the updated DABURefined algorithm
daru = DABURefined(1000, 2)  # 1000 function evaluations, 2 dimensions
print(daru(test_function))  # prints a random value between -10 and 10

# Evaluate the DABU algorithm with a refined strategy
dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10

# Evaluate the DABURefined algorithm with a novel metaheuristic strategy
daru_refined = DABURefined(1000, 2)  # 1000 function evaluations, 2 dimensions
print(daru_refined(test_function))  # prints a random value between -10 and 10