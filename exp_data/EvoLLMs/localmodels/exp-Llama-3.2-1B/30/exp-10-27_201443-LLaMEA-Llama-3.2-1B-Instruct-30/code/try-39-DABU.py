import numpy as np
from scipy.optimize import minimize

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
        return f"DABU: {self.budget} function evaluations, {self.dim} dimensions"

class AGBU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.best_func = None
        self.best_score = 0.0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        if self.func_evaluations > self.budget:
            return None
        return func_value

    def __str__(self):
        return f"AGBU: {self.budget} function evaluations, {self.dim} dimensions"

class BFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.best_func = None
        self.best_score = 0.0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Gradient-based optimization
            grad = np.zeros(self.dim)
            for i in range(self.dim):
                grad[i] = func(self.search_space[i])
            # Compute the inverse Hessian
            H = np.diag(np.abs(grad)**2)
            # Newton step
            x_new = np.linalg.inv(H).dot(grad)
            # Update the search space
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            self.func_evaluations += 1
            # Check convergence
            if np.allclose(self.search_space, x_new, atol=1e-6):
                break
        return func(self.search_space)

    def __str__(self):
        return f"BFGS: {self.budget} function evaluations, {self.dim} dimensions"

class L-BFGS-B:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.best_func = None
        self.best_score = 0.0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimization
            H = np.diag(np.abs(np.array(self.search_space) - np.array(self.search_space[0])))
            x_new = np.linalg.inv(H).dot(np.array(self.search_space) - np.array(self.search_space[0]))
            # Update the search space
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            self.func_evaluations += 1
            # Check convergence
            if np.allclose(self.search_space, x_new, atol=1e-6):
                break
        return func(self.search_space)

    def __str__(self):
        return f"L-BFGS-B: {self.budget} function evaluations, {self.dim} dimensions"

# Test the algorithms
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10

agbu = AGBU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(agbu(test_function))  # prints a random value between -10 and 10

bfgs = BFGS(1000, 2)  # 1000 function evaluations, 2 dimensions
print(bfgs(test_function))  # prints a random value between -10 and 10

lbfgs_b = L-BFGS-B(1000, 2)  # 1000 function evaluations, 2 dimensions
print(lbfgs_b(test_function))  # prints a random value between -10 and 10