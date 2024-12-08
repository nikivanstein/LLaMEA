import random
import numpy as np

class AdaptiveResamplingBBOB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()
        self.resampling_prob = 0.4
        self.line_search_prob = 0.6
        self.current_resampling = 0

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, x0, bounds, budget):
        if self.current_resampling < budget:
            return func(x0, bounds)
        else:
            x = x0
            for _ in range(budget):
                x = func(x, bounds)
                if x < bounds[0]:
                    x = bounds[0]
                elif x > bounds[1]:
                    x = bounds[1]
                if random.random() < 0.5:
                    x = random.uniform(bounds[0], bounds[1])
                if random.random() < 0.2:
                    x = random.uniform(bounds[0], bounds[1])
                if random.random() < 0.4:
                    x = random.uniform(bounds[0], bounds[1])
            return x

    def update(self, func, x0, bounds, budget):
        new_individual = self.evaluate_fitness(func, x0)
        best_individual = self.optimize(func, x0, bounds, budget)
        if new_individual > best_individual:
            x0, bounds = best_individual, new_individual
        return x0, bounds

    def evaluate_fitness(self, func, x):
        return func(x)

    def optimize(self, func, x0, bounds, budget):
        new_individual = self.__call__(func, x0, bounds, budget)
        if self.line_search_prob > random.random():
            x0, bounds = self.resample(x0, bounds)
        return new_individual

    def resample(self, x0, bounds):
        new_individual = x0
        for _ in range(self.current_resampling):
            new_individual = self.__call__(new_individual, bounds)
        return new_individual

    def update_resampling(self):
        self.current_resampling = min(self.current_resampling + self.resampling_prob, self.budget)