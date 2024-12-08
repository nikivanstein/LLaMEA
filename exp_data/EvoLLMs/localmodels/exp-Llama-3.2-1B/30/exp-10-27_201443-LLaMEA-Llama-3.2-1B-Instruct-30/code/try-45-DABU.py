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

    def __str__(self):
        return f'DABU: {self.dim}D, {self.budget} evaluations, Area over convergence curve (AOCC): {self.func_evaluations}/1000'

class AdaptiveDABU(DABU):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        # Initialize a random search space
        search_space = np.random.uniform(-5.0, 5.0, self.dim)

        # Perform 10 initial evaluations
        results = [func(x) for x in search_space]

        # Calculate the average function value
        avg_func_value = np.mean(results)

        # Refine the search space based on the average function value
        if np.abs(avg_func_value) < 1e-6:
            # If the average function value is close to zero, stop exploring the search space
            return avg_func_value
        else:
            # Otherwise, try to move the search space towards the average function value
            search_space = np.linspace(avg_func_value - 1e-6, avg_func_value + 1e-6, self.dim)
            return func(search_space)

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu = AdaptiveDABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10