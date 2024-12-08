import random
import numpy as np
from scipy.optimize import minimize

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None

    def __call__(self, func):
        if self.func is None:
            raise ValueError("func() must be called before calling __call__")

        # Initialize search space
        self.search_space = [(-5.0, 5.0)] * self.dim

        # Randomly select an initial point
        self.x0 = random.uniform(self.search_space)

        # Initialize population
        self.population = [self.x0]

        # Evaluate function at each point in population
        for _ in range(self.budget):
            # Evaluate function at each point in population
            func_values = [func(self.x) for self.x in self.population]

            # Select parents using tournament selection
            parents = []
            for _ in range(self.dim):
                for i in range(self.population):
                    if i == 0:
                        parents.append(self.population[i])
                    else:
                        idx = random.randint(0, len(self.population) - 1)
                        if func_values[i] > func_values[idx]:
                            parents.append(self.population[idx])

            # Crossover (recombination)
            self.population = []
            for _ in range(self.dim):
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                self.population.append(parents[(parent1 + parent2) // 2])

            # Mutate (perturbation)
            for i in range(self.dim):
                if random.random() < 0.25:  # Refine strategy by changing individual lines
                    self.population[i] = func(self.population[i])

        # Return best individual
        best_individual = self.population[0]
        best_func_value = func(best_individual)
        for individual in self.population:
            func_value = func(individual)
            if func_value < best_func_value:
                best_individual = individual
                best_func_value = func_value

        return best_individual, best_func_value

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization on BBOB test suite
# using tournament selection, recombination, and mutation to search for the optimal solution
# with the ability to refine strategy by changing individual lines

# Exception handling code
def handle_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Exception occurred: {e}")
            return -inf
    return wrapper

# Mutation function
def mutate(individual):
    return func(individual)

# Refine strategy function
def refine_strategy(individual):
    return mutate(individual)

# Example usage
def func1(x):
    return x[0]**2 + x[1]**2

def func2(x):
    return x[0] + x[1]

bboo_metaheuristic = BBOBMetaheuristic(100, 2)
best_individual, best_func_value = bboo_metaheuristic(func1)
print(f"Best individual: {best_individual}, Best function value: {best_func_value}")