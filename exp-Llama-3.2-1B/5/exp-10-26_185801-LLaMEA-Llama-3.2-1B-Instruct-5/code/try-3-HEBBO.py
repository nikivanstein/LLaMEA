import numpy as np
import random

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            if np.isnan(func(self.search_space)) or np.isinf(func(self.search_space)):
                raise ValueError("Invalid function value")
            if func(self.search_space) < 0 or func(self.search_space) > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func(self.search_space)

class MutationExp:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func, individual, mutation_rate):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            new_individual = individual.copy()
            if random.random() < mutation_rate:
                new_individual[random.randint(0, self.dim - 1)] = random.uniform(-5.0, 5.0)
            new_individual = np.clip(new_individual, -5.0, 5.0)
            if np.isnan(new_individual) or np.isinf(new_individual):
                raise ValueError("Invalid new individual value")
            if new_individual < 0 or new_individual > 1:
                raise ValueError("New individual value must be between 0 and 1")
            self.func_evaluations += 1
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return new_individual

class HESBBO(MutationExp):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func, individual, mutation_rate):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            new_individual = self.__call__(func, individual, mutation_rate)
            if np.isnan(new_individual) or np.isinf(new_individual):
                raise ValueError("Invalid new individual value")
            if new_individual < 0 or new_individual > 1:
                raise ValueError("New individual value must be between 0 and 1")
            self.func_evaluations += 1
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return new_individual

class HESBOB(MutationExp):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func, individual, mutation_rate):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            new_individual = self.__call__(func, individual, mutation_rate)
            if np.isnan(new_individual) or np.isinf(new_individual):
                raise ValueError("Invalid new individual value")
            if new_individual < 0 or new_individual > 1:
                raise ValueError("New individual value must be between 0 and 1")
            self.func_evaluations += 1
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return new_individual

class HESBOB2(MutationExp):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func, individual, mutation_rate):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            new_individual = self.__call__(func, individual, mutation_rate)
            if np.isnan(new_individual) or np.isinf(new_individual):
                raise ValueError("Invalid new individual value")
            if new_individual < 0 or new_individual > 1:
                raise ValueError("New individual value must be between 0 and 1")
            self.func_evaluations += 1
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return new_individual

# Example usage
if __name__ == "__main__":
    # Create a HESBBO instance
    hesbbo = HESBBO(100, 5)

    # Define a function to optimize
    def func(x):
        return x[0] + x[1] + x[2]

    # Optimize the function using HESBBO
    hsbbo = hesbbo(func, np.array([0, 0, 0]), 0.01)

    # Print the result
    print("Optimized individual:", hsbbo)