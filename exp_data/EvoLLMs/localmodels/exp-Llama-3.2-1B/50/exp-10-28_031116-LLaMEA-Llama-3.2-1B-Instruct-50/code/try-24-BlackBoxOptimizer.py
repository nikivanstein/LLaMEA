import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        # Initialize the population with random solutions
        population = []
        for _ in range(self.population_size):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population

    def __call__(self, func):
        # Evaluate the black box function with the current population
        evaluations = [func(solution) for solution in self.population]
        # Select the best solution based on the budget
        selected_solution = self.select_best_solution(evaluations, self.budget)
        # Optimize the selected solution
        return self.optimize_solution(selected_solution)

    def select_best_solution(self, evaluations, budget):
        # Select the solution with the highest value based on the budget
        selected_solution = max(evaluations, key=evaluations.index)
        # Refine the strategy to avoid exploring the same solution too much
        if len(self.population) > 1 and evaluations[-1] < evaluations[-2]:
            selected_solution = random.choice([solution for solution in self.population if solution!= selected_solution])
        return selected_solution

    def optimize_solution(self, solution):
        # Perform a random search in the search space
        best_solution = solution
        best_value = self.evaluate_function(solution)
        for _ in range(self.budget):
            # Generate a new solution by perturbing the current solution
            perturbed_solution = best_solution + random.uniform(-1.0, 1.0, self.dim)
            # Evaluate the new solution
            new_value = self.evaluate_function(perturbed_solution)
            # Update the best solution if a better one is found
            if new_value > best_value:
                best_solution = perturbed_solution
                best_value = new_value
        return best_solution, best_value

    def evaluate_function(self, solution):
        # Evaluate the black box function at the current solution
        return np.mean([func(solution)] * self.dim)

# Test the algorithm
budget = 100
dim = 10
optimizer = BlackBoxOptimizer(budget, dim)
func = lambda x: x**2
optimizer(func)