import random
import numpy as np

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.generate_population()

    def generate_population(self):
        # Initialize population with random solutions
        return [[random.uniform(-5.0, 5.0) for _ in range(self.dim)] for _ in range(self.population_size)]

    def __call__(self, func, num_evaluations):
        # Evaluate function for each solution in the population
        solutions = []
        for solution in self.population:
            func_value = func(solution)
            solutions.append((solution, func_value))
            if len(solutions) >= num_evaluations:
                break

        # Select the best solution based on the budget
        best_solution = max(solutions, key=lambda x: x[1])

        # Refine the solution based on the budget
        if len(solutions) < self.budget:
            for _ in range(self.budget - len(solutions)):
                # Use the current solution as a starting point and refine it using the probability 0.45
                new_solution = self.refine_solution(best_solution, 0.45)
                best_solution = new_solution

        # Evaluate the best solution
        func_value = func(best_solution)
        return best_solution, func_value

    def refine_solution(self, solution, probability):
        # Refine the solution using the probability 0.45
        new_solution = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
        for i in range(self.dim):
            new_solution[i] += random.uniform(-0.1, 0.1)
            new_solution[i] = max(-5.0, min(new_solution[i], 5.0))
            new_solution[i] = max(-5.0, min(new_solution[i], 5.0))
            new_solution[i] = max(-5.0, min(new_solution[i], 5.0))
        return new_solution

# Define the black box function
def func(x):
    return np.sum(x**2)

# Create an instance of the Metaheuristic algorithm
metaheuristic = Metaheuristic(100, 5)

# Optimize the function using the Metaheuristic algorithm
best_solution, best_func_value = metaheuristic(__call__, 100)
print(f"Best solution: {best_solution}, Best function value: {best_func_value}")