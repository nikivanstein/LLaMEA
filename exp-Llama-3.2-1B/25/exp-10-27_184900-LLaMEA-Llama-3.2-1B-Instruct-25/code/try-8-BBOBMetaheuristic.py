import random
import numpy as np
from copy import deepcopy
from operator import itemgetter

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
                self.population.append(deepcopy(parents[(parent1 + parent2) // 2]))

            # Mutate (perturbation)
            for i in range(self.dim):
                if random.random() < 0.25:
                    self.population[i] = random.uniform(-1.0, 1.0)

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

class BBOBMetaheuristicAlgorithm:
    def __init__(self, budget, dim, mutation_rate=0.25):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
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
                self.population.append(deepcopy(parents[(parent1 + parent2) // 2]))

            # Mutate (perturbation)
            for i in range(self.dim):
                if random.random() < self.mutation_rate:
                    self.population[i] = random.uniform(-1.0, 1.0)

        # Return best individual
        best_individual = self.population[0]
        best_func_value = func(best_individual)
        for individual in self.population:
            func_value = func(individual)
            if func_value < best_func_value:
                best_individual = individual
                best_func_value = func_value

        return best_individual, best_func_value

# Test the algorithm
def test_bbb():
    # Define the BBOB test suite
    test_suite = [
        {'name': 'f1', 'func': lambda x: x**2 + 2*x + 1},
        {'name': 'f2', 'func': lambda x: x**2 - 2*x + 1},
        {'name': 'f3', 'func': lambda x: x**2 + 3*x + 2},
        {'name': 'f4', 'func': lambda x: x**2 - 3*x + 4},
        {'name': 'f5', 'func': lambda x: x**2 + 5*x + 5},
    ]

    # Initialize the algorithm
    algorithm = BBOBMetaheuristicAlgorithm(budget=10, dim=5)

    # Evaluate the function at each point in the population
    for test_case in test_suite:
        func = test_case['func']
        best_individual, best_func_value = algorithm(func)

        # Print the results
        print(f"Test case: {test_case['name']}")
        print(f"Best individual: {best_individual}")
        print(f"Best function value: {best_func_value}")
        print()

# Run the test
test_bbb()