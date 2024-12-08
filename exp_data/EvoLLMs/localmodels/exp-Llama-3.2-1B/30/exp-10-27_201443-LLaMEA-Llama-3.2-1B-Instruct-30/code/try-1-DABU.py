import numpy as np
import random

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
        return f'DABU: {self.dim}D: {self.budget} evaluations'

class DABU2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.search_history = []

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
            self.search_history.append(self.search_space)
        return func_value

    def __str__(self):
        return f'DABU2: {self.dim}D: {self.budget} evaluations'

class DABU3:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.search_history = []
        self.covariance = np.array([[1, 0.5], [0.5, 1]])

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
            self.search_history.append(self.search_space)
        return func_value

    def __str__(self):
        return f'DABU3: {self.dim}D: {self.budget} evaluations'

class DABU4:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.search_history = []
        self.covariance = np.array([[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]])

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
            self.search_history.append(self.search_space)
        return func_value

    def __str__(self):
        return f'DABU4: {self.dim}D: {self.budget} evaluations'

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu2 = DABU2(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu2(test_function))  # prints a random value between -10 and 10

dabu3 = DABU3(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu3(test_function))  # prints a random value between -10 and 10

dabu4 = DABU4(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu4(test_function))  # prints a random value between -10 and 10