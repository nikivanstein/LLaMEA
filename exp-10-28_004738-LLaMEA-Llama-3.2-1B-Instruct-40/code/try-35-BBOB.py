# Description: Adaptive Multi-Step Optimization using Adaptive Step Size and Adaptive Mutation
# Code: 
import random
import math

class BBOB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()
        self.step_sizes = [5.0] * self.dim
        self.mutations = [True] * self.dim
        self.step_size_history = [5.0] * self.dim

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, x0, bounds, budget):
        for _ in range(budget):
            x = x0
            for i in range(self.dim):
                if random.random() < 0.4:
                    x = func(x)
                    if x < bounds[i]:
                        x = bounds[i]
                    elif x > bounds[i]:
                        x = bounds[i]
                    if random.random() < 0.2:
                        x = random.uniform(bounds[i])
                    if random.random() < 0.4:
                        x = random.uniform(bounds[i])
                    if random.random() < 0.4:
                        x = random.uniform(bounds[i])
            if random.random() < 0.2:
                x = random.uniform(bounds[i])
            if random.random() < 0.4:
                x = random.uniform(bounds[i])
            if random.random() < 0.4:
                x = random.uniform(bounds[i])
            if random.random() < 0.4:
                x = random.uniform(bounds[i])
            new_individual = x
            self.step_sizes[i] = func(new_individual)
            self.mutations[i] = random.random() < 0.2
            self.step_size_history[i] = self.step_sizes[i]
            if self.step_size_history[i] < self.step_sizes[i]:
                self.step_size_history[i] = self.step_sizes[i]
            if self.step_size_history[i] < 2.5:
                self.step_size_history[i] = 2.5
            updated_individual = self.evaluate_fitness(new_individual)
            updated_individual = self.f(updated_individual, self.step_size_history[i])
            new_individual = updated_individual
            self.step_sizes[i] = self.step_size_history[i]
            self.mutations[i] = self.mutations[i]
            self.step_size_history[i] = self.step_size_history[i]
            if self.step_size_history[i] < self.step_sizes[i]:
                self.step_size_history[i] = self.step_sizes[i]
            if self.step_size_history[i] < 2.5:
                self.step_size_history[i] = 2.5
            if updated_individual < bounds[i]:
                new_individual = bounds[i]
            elif updated_individual > bounds[i]:
                new_individual = bounds[i]
            if random.random() < 0.2:
                new_individual = random.uniform(bounds[i])
            if random.random() < 0.2:
                new_individual = random.uniform(bounds[i])
            if random.random() < 0.2:
                new_individual = random.uniform(bounds[i])
            if random.random() < 0.2:
                new_individual = random.uniform(bounds[i])
            updated_individual = self.f(new_individual, self.step_size_history[i])
            updated_individual = updated_individual
            new_individual = updated_individual
            self.step_sizes[i] = self.step_size_history[i]
            self.mutations[i] = self.mutations[i]
            self.step_size_history[i] = self.step_size_history[i]
            if self.step_size_history[i] < self.step_sizes[i]:
                self.step_size_history[i] = self.step_sizes[i]
            if self.step_size_history[i] < 2.5:
                self.step_size_history[i] = 2.5
            return new_individual

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget):
    return BBOB(budget, len(bounds)).__call__(func, x0, bounds, budget)

# Description: Black Box Optimization using BBOB
# Code: 