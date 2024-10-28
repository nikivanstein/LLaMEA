import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        return [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

    def __call__(self, func):
        def evaluate_func(x):
            return func(x)

        for _ in range(self.budget):
            best_x = None
            best_score = -np.inf
            for x in self.population:
                score = evaluate_func(x)
                if score > best_score:
                    best_x = x
                    best_score = score
            self.population.remove(best_x)
            self.population.append(best_x)

        return self.population

    def select_solution(self, func):
        def evaluate_func(x):
            return func(x)

        best_x = None
        best_score = -np.inf
        for _ in range(self.budget):
            score = evaluate_func(np.random.uniform(-5.0, 5.0, self.dim))
            if score > best_score:
                best_x = np.random.uniform(-5.0, 5.0, self.dim)
                best_score = score

        return best_x

    def mutate_solution(self, func, x):
        def evaluate_func(x):
            return func(x)

        best_x = np.random.uniform(-5.0, 5.0, self.dim)
        for _ in range(100):
            score = evaluate_func(best_x)
            if score > evaluate_func(x):
                best_x = x

        return best_x

    def crossover(self, parent1, parent2):
        def evaluate_func(x):
            return func(x)

        best_x1 = np.random.uniform(-5.0, 5.0, self.dim)
        best_x2 = np.random.uniform(-5.0, 5.0, self.dim)

        for _ in range(100):
            score1 = evaluate_func(best_x1)
            score2 = evaluate_func(best_x2)
            if score1 > score2:
                best_x1 = x
            else:
                best_x2 = x

        return best_x1, best_x2

    def train(self):
        best_solution = None
        best_score = -np.inf

        for _ in range(1000):
            func = random.choice([func for func in BlackBoxOptimizer.budget])
            solution = self.__call__(func)
            score = evaluate_func(solution)
            if score > best_score:
                best_solution = solution
                best_score = score

        self.population = self.population[:best_solution]
        self.best_solution = best_solution
        self.best_score = best_score

        return best_solution, best_score

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using genetic algorithm with mutation and crossover.
# The algorithm evaluates a set of noiseless functions with a given budget and dimensionality.
# It uses a population of random solutions and evolves them using genetic operations (mutation and crossover) to find the optimal solution.