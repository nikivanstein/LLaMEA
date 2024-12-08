import numpy as np

class MetaGradientDescent:
    def __init__(self, budget, dim, learning_rate, alpha, tol):
        self.budget = budget
        self.dim = dim
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.tol = tol
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            gradient = np.gradient(func(x), axis=0)
            learning_rate = self.learning_rate * self.alpha / np.sqrt(self.func_evaluations)
            new_x = x - learning_rate * np.dot(gradient, x)
            self.func_evaluations += 1
            try:
                result = minimize(wrapper, new_x, method="SLSQP", bounds=[(x, x) for x in self.search_space])
                return result.x
            except Exception as e:
                print(f"Error: {e}")
                return None

        return wrapper