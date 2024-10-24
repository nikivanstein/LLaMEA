import numpy as np
from scipy.optimize import minimize
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.initialize_population()

    def initialize_population(self):
        # Initialize population with random solutions
        population = []
        for _ in range(100):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population

    def __call__(self, func):
        # Evaluate the black box function with the current population
        evaluations = [func(solution) for solution in self.population]
        # Select the best solution based on the budget
        selected_solution = self.select_best_solution(evaluations, self.budget)
        # Optimize the selected solution
        return selected_solution

    def select_best_solution(self, evaluations, budget):
        # Use a novel heuristic to select the best solution
        # In this case, we use a weighted average of the evaluations
        weights = np.array([1.0 / i for i in range(1, budget + 1)])
        selected_solution = np.random.choice(self.population, p=weights)
        return selected_solution

    def mutate(self, solution):
        # Randomly mutate a solution
        mutated_solution = solution.copy()
        if random.random() < 0.1:
            mutated_solution[random.randint(0, self.dim - 1)] = random.uniform(-5.0, 5.0)
        return mutated_solution