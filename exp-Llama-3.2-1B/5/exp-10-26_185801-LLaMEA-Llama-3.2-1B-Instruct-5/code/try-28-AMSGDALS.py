import numpy as np

class AMSGDALS:
    def __init__(self, budget, dim, learning_rate=0.01, max_iter=100, tol=1e-6):
        self.budget = budget
        self.dim = dim
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            if func_evaluations > self.max_iter:
                break
        return func_value

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

# Initialize AMSGDALS algorithm
amsgd = AMSGDALS(1000, 10)

# Initialize HEBBO algorithm
hebbbo = HEBBO(1000, 10)

# Evaluate black box function with AMSGDALS
func = lambda x: np.sin(x)
result = amsgd(hebbbo(func))

# Print result
print("Result:", result)

# Print HEBBO's solution
print("HEBBO's solution:", hebbbo(hebbbo(func)))