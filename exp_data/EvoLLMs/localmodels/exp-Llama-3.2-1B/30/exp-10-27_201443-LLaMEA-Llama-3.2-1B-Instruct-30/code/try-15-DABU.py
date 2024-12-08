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

    def adapt_search_space(self, func, initial_search_space):
        # Calculate the best function value and corresponding search space
        best_func_value = np.min([func(x) for x in initial_search_space])
        best_search_space = initial_search_space[np.argmin([func(x) for x in initial_search_space])]

        # Refine the search space based on the best function value
        if np.abs(best_func_value) > 1e-6:
            # If the best function value is significantly different from the minimum, refine the search space
            step_size = 0.1
            for i in range(self.dim):
                new_search_space = initial_search_space.copy()
                new_search_space[i] += step_size
                new_search_space = np.clip(new_search_space, best_search_space - 2*step_size, best_search_space + 2*step_size)
                self.search_space = new_search_space
                func_value = func(new_search_space)
                if np.abs(func_value) < 1e-6:
                    break
            self.search_space = np.clip(self.search_space, best_search_space - 2*step_size, best_search_space + 2*step_size)
        return self.search_space

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
initial_search_space = np.linspace(-5.0, 5.0, 2)
dabu.search_space = dabu.adapt_search_space(test_function, initial_search_space)
print(dabu(test_function))  # prints a random value between -10 and 10

# DABU's adaptive search space refinement strategy:
# 1. Calculate the best function value and corresponding search space
# 2. If the best function value is significantly different from the minimum, refine the search space
# 3. Repeat steps 1-2 until the budget is exhausted