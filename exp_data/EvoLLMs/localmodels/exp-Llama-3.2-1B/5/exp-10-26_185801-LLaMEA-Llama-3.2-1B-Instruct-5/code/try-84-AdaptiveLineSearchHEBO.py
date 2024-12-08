import numpy as np
import random
import math

class AdaptiveLineSearchHEBO:
    def __init__(self, budget, dim, alpha=0.01, mu=0.1, beta=0.9):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.alpha = alpha
        self.mu = mu
        self.beta = beta
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            new_individual = self.evaluate_fitness(func_value)
            if func_value <= new_individual:
                new_individual = func_value
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            if random.random() < self.beta:
                new_individual = self.optimize_func(func_value, new_individual)
            self.search_space = new_individual
        return func_value

    def evaluate_fitness(self, func_value):
        updated_individual = func_value
        while updated_individual < func_value:
            updated_individual = self.f(updated_individual, self.logger)
        return updated_individual

    def optimize_func(self, func_value, new_individual):
        # Use adaptive line search to find the optimal function value
        # This is a novel heuristic algorithm that uses a combination of
        # gradient descent and adaptive line search to find the optimal function
        # value.
        d_f = self.f_derivative(func_value)
        if d_f == 0:
            return new_individual
        step_size = self.alpha * d_f
        new_individual = new_individual - step_size
        return new_individual

    def f_derivative(self, func_value):
        # This is a simple implementation of the gradient descent
        # algorithm for finding the derivative of a function.
        # In this case, we assume that the function is a simple
        # linear function.
        return (func_value - 0.5) / 0.1

    def f(self, individual, logger):
        # This is a simple implementation of the function
        # that we want to optimize.
        return individual * 0.1 + random.random() * 0.5

    def logger(self):
        # This is a simple implementation of a logger that
        # prints the current fitness value to the console.
        print(f"Current fitness: {self.func_evaluations / self.budget:.2f}")