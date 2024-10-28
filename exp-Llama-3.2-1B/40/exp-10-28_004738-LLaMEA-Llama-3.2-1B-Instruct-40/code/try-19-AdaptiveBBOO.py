import random
import numpy as np

class AdaptiveBBOO:
    def __init__(self, budget, dim, learning_rate, decay_rate, max_iter):
        self.budget = budget
        self.dim = dim
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.max_iter = max_iter
        self.funcs = self.generate_functions()
        self.func_history = []
        self.new_individual = None

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: np.random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, x0, bounds, iterations=1000):
        for _ in range(iterations):
            x = x0
            for _ in range(self.budget):
                x = func(x)
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
            self.func_history.append(x)
            if len(self.func_history) > 10:
                x = self.func_history[-1]
                if x < bounds[0] and random.random() < 0.4:
                    x = bounds[0]
                elif x > bounds[1] and random.random() < 0.4:
                    x = bounds[1]
                if random.random() < 0.4:
                    x = random.uniform(bounds[0], bounds[1])
                if random.random() < 0.2:
                    x = random.uniform(bounds[0], bounds[1])
                if random.random() < 0.4:
                    x = random.uniform(bounds[0], bounds[1])
            return x

    def evaluate_fitness(self, func, x):
        return func(x)

    def mutate(self, x):
        if random.random() < self.decay_rate:
            return x + random.uniform(-1, 1)
        return x

    def __str__(self):
        return f"AdaptiveBBOO(budget={self.budget}, dim={self.dim}, learning_rate={self.learning_rate}, decay_rate={self.decay_rate}, max_iter={self.max_iter})"

# Usage:
adaptive_bboo = AdaptiveBBOO(budget=1000, dim=5, learning_rate=0.01, decay_rate=0.2, max_iter=1000)
adaptive_bboo_func = lambda x: x**2 + 0.5*x + 0.1
adaptive_bboo.new_individual = adaptive_bboo_func(random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), [[-5.0, -5.0], [5.0, 5.0]])
adaptive_bboo(new_individual = adaptive_bboo_func(random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), [[-5.0, -5.0], [5.0, 5.0]]))

# Print the current solution
print(adaptive_bboo)

# Print the fitness of the current solution
print(adaptive_bboo.evaluate_fitness(adaptive_bboo_func, adaptive_bboo.new_individual))

# Optimize the solution using the AdaptiveBBOO algorithm
optimal_individual = adaptive_bboo(new_individual=adaptive_bboo_func(random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), [[-5.0, -5.0], [5.0, 5.0]]))
print(optimal_individual)
print(adaptive_bboo.evaluate_fitness(optimal_individual, optimal_individual))