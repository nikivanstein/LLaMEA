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

class DASO:
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
        return f"DASO: {self.budget} function evaluations, {self.dim} dimensions"

class ADAM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.alpha = 0.9
        self.beta = 0.999
        self.moving_average = np.zeros(self.dim)
        self.variance = np.zeros(self.dim)

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            self.variance = np.mean((self.variance * self.beta + self.moving_average * (1 - self.beta)) ** 2)
            self.moving_average = self.beta * self.moving_average + (1 - self.beta) * func_value
            self.variance = self.beta * self.variance + (1 - self.beta) * self.variance * self.variance
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def __str__(self):
        return f"ADAM: {self.budget} function evaluations, {self.dim} dimensions"

class PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.particles = []
        self.vectors = []

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def __str__(self):
        return f"PSO: {self.budget} function evaluations, {self.dim} dimensions"

class NSGAII:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = []
        self.fitness = np.zeros(self.budget)
        self.pareto = False

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            self.func_evaluations += 1
        return func_value

    def __str__(self):
        return f"NSGAII: {self.budget} function evaluations, {self.dim} dimensions"

# Description: Genetic Algorithm
# Code: 