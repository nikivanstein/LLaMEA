import numpy as np

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

class HookeJenks:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            new_individual = np.copy(self.search_space)
            for i in range(self.dim):
                for j in range(self.dim):
                    if np.random.rand() < 0.5:
                        new_individual[i] += np.random.uniform(-1, 1)
                        new_individual[j] += np.random.uniform(-1, 1)
            new_individual = np.clip(new_individual, self.search_space, None)
            func_value = func(new_individual)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

class NSGAII:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            new_individual = np.copy(self.search_space)
            for i in range(self.dim):
                for j in range(self.dim):
                    if np.random.rand() < 0.5:
                        new_individual[i] += np.random.uniform(-1, 1)
                        new_individual[j] += np.random.uniform(-1, 1)
            new_individual = np.clip(new_individual, self.search_space, None)
            func_value = func(new_individual)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

class NSGAII2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            new_individual = np.copy(self.search_space)
            for i in range(self.dim):
                for j in range(self.dim):
                    if np.random.rand() < 0.5:
                        new_individual[i] += np.random.uniform(-1, 1)
                        new_individual[j] += np.random.uniform(-1, 1)
            new_individual = np.clip(new_individual, self.search_space, None)
            func_value = func(new_individual)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

# Example usage
func1 = lambda x: np.sin(x)
func2 = lambda x: x**2

hestbo = HEBBO(budget=100, dim=2)
hestbo(func1)

hestbo2 = HookeJenks(budget=100, dim=2)
hestbo2(func1)

hestbo3 = NSGAII(budget=100, dim=2)
hestbo3(func1)

hestbo4 = NSGAII2(budget=100, dim=2)
hestbo4(func1)