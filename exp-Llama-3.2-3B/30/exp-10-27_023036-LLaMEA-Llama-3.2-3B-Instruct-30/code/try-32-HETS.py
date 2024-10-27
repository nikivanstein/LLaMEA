import numpy as np
import random
from scipy.optimize import minimize

class HETS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.tabu_list = []
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.population = [self.search_space[i] for i in random.sample(range(dim), population_size)]

    def __call__(self, func):
        for _ in range(self.budget):
            best_solution = self.population[np.argmin([func(x) for x in self.population])]
            score = func(best_solution)
            self.population = [self.search_space[i] for i in random.sample(range(dim), population_size)]
            self.population = [self.search_space[i] if random.random() > 0.3 else best_solution for i in range(dim)]
            self.population = [min(max(x, -5.0), 5.0) for x in self.population]
            self.population = [x for x in self.population if x not in self.tabu_list]
            self.tabu_list.extend([best_solution] + [x for x in self.population if x not in self.tabu_list])
            self.tabu_list = self.tabu_list[-self.population_size:]
            if score < np.inf:
                print(f"New best solution: {best_solution}, Score: {score}")

def evaluate_function(func, x):
    return func(x)

def optimize_function(func, hets, dim):
    x0 = np.linspace(-5.0, 5.0, dim)
    res = minimize(evaluate_function(func, x0), x0, method="SLSQP")
    return res.x

# Example usage:
def example_func(x):
    return x[0]**2 + x[1]**2

hets = HETS(budget=100, dim=2)
hets(example_func)