import random
import numpy as np

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class Mutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, individual, new_individual):
        # Refine the strategy by changing the lines of the selected solution
        if random.random() < 0.45:
            # Randomly select a line from the selected solution
            line = random.randint(0, len(individual) - 1)

            # Refine the strategy by changing the line
            new_individual[line] = (individual[line] + random.uniform(-0.1, 0.1)) / 2

        return new_individual

class Selection:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func, population):
        # Select the best individual from the population
        return max(set(population), key=func)

class Optimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func, population, mutation, selection):
        # Initialize the population with the selected solution
        population = [self.select_solution(func, population, self.budget, self.dim)]

        # Run the optimization algorithm for a fixed number of iterations
        for _ in range(100):
            # Evaluate the function a limited number of times
            num_evals = min(self.budget, len(func(self.search_space)))
            func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

            # Select the best function value
            best_func = max(set(func_values), key=func_values.count)

            # Update the search space
            self.search_space = [x for x in self.search_space if x not in best_func]

            # Apply mutation
            mutation.apply(self, population, func)

            # Select the new population
            population = [self.select_solution(func, population, self.budget, self.dim) for _ in range(self.budget)]

        return population

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        return Metaheuristic(self.budget, self.dim).__call__(func)

class BBOB:
    def __init__(self, problem):
        self.problem = problem
        self.population = [self.select_solution(func, [self.problem], self.budget, self.problem.dim) for _ in range(100)]

    def select_solution(self, func, population, budget, dim):
        return Selection(budget, dim).__call__(func, population)

    def apply(self, algorithm, population, func):
        return Optimization(budget, dim).__call__(func, population, algorithm.mutation, algorithm.selection)

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 