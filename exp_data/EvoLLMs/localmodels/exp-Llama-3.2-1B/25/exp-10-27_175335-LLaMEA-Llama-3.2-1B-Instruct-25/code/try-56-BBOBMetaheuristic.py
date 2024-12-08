import numpy as np
import random

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.best_individual = None

    def __call__(self, func):
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func):
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        sol = None
        for _ in range(10):
            sol = np.random.uniform(bounds, size=self.dim)
            func_sol = self.__call__(func, sol)
            if func_sol < self.__call__(func, sol):
                sol = sol
        return sol

    def mutate(self, individual):
        if self.best_individual is None:
            self.best_individual = individual
        else:
            if random.random() < 0.25:
                new_individual = individual + np.random.uniform(-1, 1, self.dim)
                if new_individual < -5.0:
                    new_individual = -5.0
                elif new_individual > 5.0:
                    new_individual = 5.0
                if new_individual not in self.best_individual:
                    self.best_individual = new_individual
        return new_individual

    def evaluate_fitness(self, individual):
        func = self.__call__(self.func, individual)
        updated_individual = self.search(func)
        aucs = self.func_evals / self.budget
        updated_fitness = func(updated_individual)
        return updated_fitness, aucs

    def __str__(self):
        return f"Evolutionary Algorithm for Black Box Optimization using Genetic Programming"

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 