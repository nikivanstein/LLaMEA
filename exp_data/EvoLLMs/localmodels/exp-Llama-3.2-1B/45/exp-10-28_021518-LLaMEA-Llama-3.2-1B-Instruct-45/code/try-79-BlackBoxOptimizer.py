import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.elite_size = 10
        self.population = self.generate_population()
        self.elite = self.population[:self.elite_size]

    def generate_population(self):
        population = []
        for _ in range(self.population_size):
            dim = self.dim
            for _ in range(dim):
                population.append(np.random.uniform(-5.0, 5.0))
        return population

    def __call__(self, func):
        def evaluate_func(x):
            return func(x)

        def fitness_func(x):
            return evaluate_func(x)

        while len(self.elite) < self.elite_size:
            # Selection
            fitness_values = [fitness_func(x) for x in self.population]
            indices = np.argsort(fitness_values)[:self.population_size]
            self.elite = [self.population[i] for i in indices]

            # Crossover
            children = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(self.elite, 2)
                child = (parent1 + parent2) / 2
                children.append(child)

            # Mutation
            for child in children:
                if random.random() < 0.1:
                    index = random.randint(0, self.dim - 1)
                    child[index] += random.uniform(-1.0, 1.0)

            # Replace the elite with the children
            self.elite = children

        # Adaptive Line Search
        def adaptive_line_search(func, x0, bounds, tol=1e-2, max_iter=100):
            x = x0
            for _ in range(max_iter):
                res = func(x)
                if res < tol:
                    return x
                f_x = func(x)
                if f_x < tol:
                    return x
                # Compute the gradient
                gradient = np.zeros(self.dim)
                for i in range(self.dim):
                    gradient[i] = (func(x + 0.1 * gradient[i]) - func(x - 0.1 * gradient[i])) / 2.0
                # Update the direction
                direction = gradient / np.linalg.norm(gradient)
                # Update the step size
                step_size = 0.1 * np.linalg.norm(direction)
                # Update the position
                x = x + step_size * direction
            return x

        adaptive_line_search_func = lambda x: evaluate_func(adaptive_line_search(func, x, bounds))

        # Replace the elite with the children using Adaptive Line Search
        self.elite = adaptive_line_search_func(self.elite[0])

        return self.elite[0]