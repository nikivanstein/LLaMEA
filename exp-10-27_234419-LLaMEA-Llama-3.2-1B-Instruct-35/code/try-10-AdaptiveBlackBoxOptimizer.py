# Description: Adaptive Black Box Optimization using Multi-Objective Genetic Algorithm
# Code: 
# ```python
import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.sample_size = None
        self.sample_indices = None
        self.local_search = False
        self.best_individual = None
        self.best_fitness = float('-inf')

    def __call__(self, func):
        if self.func is None:
            self.func = func
            self.search_space = np.random.uniform(-5.0, 5.0, self.dim)
            self.sample_size = 1
            self.sample_indices = None

        if self.budget <= 0:
            raise ValueError("Budget is less than or equal to zero")

        for _ in range(self.budget):
            if self.sample_indices is None:
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            else:
                self.sample_indices = np.random.choice(self.sample_indices, size=self.sample_size, replace=False)
            self.local_search = False

            if self.local_search:
                best_func = func(self.sample_indices)
                if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
                    self.sample_indices = None
                    self.local_search = False
                    self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                    self.sample_indices = self.sample_indices[:self.sample_size]
                else:
                    self.sample_indices = None
                    self.local_search = False

            if self.sample_indices is None:
                best_func = func(self.sample_indices)
                self.sample_indices = None
                self.local_search = False

            if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                break

        self.best_individual = self.sample_indices
        self.best_fitness = np.abs(best_func - func(self.best_individual))

        return func(self.best_individual)

class AdaptiveBlackBoxOptimizerWithAdaptiveSampling(AdaptiveBlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.sample_indices = None
        self.sample_size = None

    def __call__(self, func):
        if self.sample_indices is None:
            self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            self.sample_size = 1
        else:
            self.sample_indices = np.random.choice(self.sample_indices, size=self.sample_size, replace=False)
            self.sample_size = 1

        best_func = func(self.sample_indices)
        self.sample_indices = None
        self.sample_size = None

        if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
            self.sample_indices = None
            self.sample_size = None
            self.local_search = False
            self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            self.sample_indices = self.sample_indices[:self.sample_size]
        else:
            self.sample_indices = None
            self.sample_size = None
            self.local_search = False

        if np.abs(best_func - func(self.sample_indices)) < 1e-6:
            return func(self.sample_indices)

        self.best_individual = self.sample_indices
        self.best_fitness = np.abs(best_func - func(self.best_individual))

        return func(self.best_individual)

class AdaptiveBlackBoxOptimizerWithLocalSearch(AdaptiveBlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search = False

    def __call__(self, func):
        if self.local_search:
            best_func = func(self.sample_indices)
            self.sample_indices = None
            self.local_search = False

            if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
                self.sample_indices = None
                self.local_search = False
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                self.sample_indices = self.sample_indices[:self.sample_size]
            else:
                self.sample_indices = None
                self.local_search = False

            if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                return func(self.sample_indices)

            self.best_individual = self.sample_indices
            self.best_fitness = np.abs(best_func - func(self.best_individual))

            return func(self.best_individual)

class AdaptiveBlackBoxOptimizerWithRandomSampling(AdaptiveBlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.sample_indices = None
        self.sample_size = None

    def __call__(self, func):
        if self.sample_indices is None:
            self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            self.sample_size = 1
        else:
            self.sample_indices = np.random.choice(self.sample_indices, size=self.sample_size, replace=False)
            self.sample_size = 1

        best_func = func(self.sample_indices)
        self.sample_indices = None
        self.sample_size = None

        if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
            self.sample_indices = None
            self.sample_size = None
            self.local_search = False
            self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            self.sample_indices = self.sample_indices[:self.sample_size]
        else:
            self.sample_indices = None
            self.sample_size = None
            self.local_search = False

        if np.abs(best_func - func(self.sample_indices)) < 1e-6:
            return func(self.sample_indices)

        self.best_individual = self.sample_indices
        self.best_fitness = np.abs(best_func - func(self.best_individual))

        return func(self.best_individual)

# Description: Adaptive Black Box Optimization using Multi-Objective Genetic Algorithm
# Code: 
# ```python
# Adaptive Black Box Optimization using Multi-Objective Genetic Algorithm
# ```
# ```python
# import numpy as np
# import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.sample_size = None
        self.sample_indices = None
        self.local_search = False
        self.best_individual = None
        self.best_fitness = float('-inf')

    def __call__(self, func):
        if self.func is None:
            self.func = func
            self.search_space = np.random.uniform(-5.0, 5.0, self.dim)
            self.sample_size = 1
            self.sample_indices = None

        if self.budget <= 0:
            raise ValueError("Budget is less than or equal to zero")

        for _ in range(self.budget):
            if self.sample_indices is None:
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            else:
                self.sample_indices = np.random.choice(self.sample_indices, size=self.sample_size, replace=False)
            self.local_search = False

            if self.local_search:
                best_func = func(self.sample_indices)
                if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
                    self.sample_indices = None
                    self.local_search = False
                    self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                    self.sample_indices = self.sample_indices[:self.sample_size]
                else:
                    self.sample_indices = None
                    self.local_search = False

            if self.sample_indices is None:
                best_func = func(self.sample_indices)
                self.sample_indices = None
                self.local_search = False

            if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                break

        self.best_individual = self.sample_indices
        self.best_fitness = np.abs(best_func - func(self.best_individual))

        return func(self.best_individual)

class AdaptiveBlackBoxOptimizerWithAdaptiveSampling(AdaptiveBlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.sample_indices = None
        self.sample_size = None

    def __call__(self, func):
        if self.sample_indices is None:
            self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            self.sample_size = 1
        else:
            self.sample_indices = np.random.choice(self.sample_indices, size=self.sample_size, replace=False)
            self.sample_size = 1

        best_func = func(self.sample_indices)
        self.sample_indices = None
        self.sample_size = None

        if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
            self.sample_indices = None
            self.sample_size = None
            self.local_search = False
            self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            self.sample_indices = self.sample_indices[:self.sample_size]
        else:
            self.sample_indices = None
            self.sample_size = None
            self.local_search = False

        if np.abs(best_func - func(self.sample_indices)) < 1e-6:
            return func(self.sample_indices)

        self.best_individual = self.sample_indices
        self.best_fitness = np.abs(best_func - func(self.best_individual))

        return func(self.best_individual)

class AdaptiveBlackBoxOptimizerWithLocalSearch(AdaptiveBlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search = False

    def __call__(self, func):
        if self.local_search:
            best_func = func(self.sample_indices)
            self.sample_indices = None
            self.local_search = False

            if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
                self.sample_indices = None
                self.local_search = False
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                self.sample_indices = self.sample_indices[:self.sample_size]
            else:
                self.sample_indices = None
                self.local_search = False

            if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                return func(self.sample_indices)

            self.best_individual = self.sample_indices
            self.best_fitness = np.abs(best_func - func(self.best_individual))

            return func(self.best_individual)

class AdaptiveBlackBoxOptimizerWithRandomSampling(AdaptiveBlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.sample_indices = None
        self.sample_size = None

    def __call__(self, func):
        if self.sample_indices is None:
            self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            self.sample_size = 1
        else:
            self.sample_indices = np.random.choice(self.sample_indices, size=self.sample_size, replace=False)
            self.sample_size = 1

        best_func = func(self.sample_indices)
        self.sample_indices = None
        self.sample_size = None

        if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
            self.sample_indices = None
            self.sample_size = None
            self.local_search = False
            self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            self.sample_indices = self.sample_indices[:self.sample_size]
        else:
            self.sample_indices = None
            self.sample_size = None
            self.local_search = False

        if np.abs(best_func - func(self.sample_indices)) < 1e-6:
            return func(self.sample_indices)

        self.best_individual = self.sample_indices
        self.best_fitness = np.abs(best_func - func(self.best_individual))

        return func(self.best_individual)

# Description: Adaptive Black Box Optimization using Multi-Objective Genetic Algorithm
# Code: 
# ```python
# Adaptive Black Box Optimization using Multi-Objective Genetic Algorithm
# ```
# ```python
# import numpy as np
# import random

def evaluate_fitness(individual, func):
    return func(individual)

def adaptive_black_box_optimizer(budget, dim):
    return AdaptiveBlackBoxOptimizer(budget, dim)

# Create an instance of the adaptive black box optimizer
optimizer = adaptive_black_box_optimizer(100, 10)

# Define a function to be optimized
def func(individual):
    return individual * individual

# Evaluate the function using the optimizer
fitness = evaluate_fitness(optimizer.best_individual, func)

# Print the result
print(f"Optimized individual: {optimizer.best_individual}")
print(f"Optimized fitness: {fitness}")