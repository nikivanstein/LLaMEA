import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.init_population()
        self.population_history = []

    def init_population(self):
        # Initialize the population with random solutions
        return [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

    def __call__(self, func):
        # Evaluate the function for each solution in the population
        solutions = [func(solution) for solution in self.population]
        # Select the top solutions based on their fitness (function value)
        top_solutions = random.sample(solutions, self.budget)
        # Refine the population using the selected solutions
        self.population = [func(solution) for solution in top_solutions]
        self.population_history.append((self.population, self.budget))

    def select_solution(self):
        # Select a random solution from the current population
        return random.choice(self.population)

    def mutate_solution(self, solution):
        # Randomly change one element in the solution
        idx = random.randint(0, self.dim - 1)
        solution[idx] = random.uniform(-5.0, 5.0)
        return solution

    def crossover(self, parent1, parent2):
        # Select a random crossover point and create a new child
        crossover_point = random.randint(0, self.dim - 1)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def __str__(self):
        return f"Population: {self.population}, Population History: {self.population_history}"

# Example usage
budget = 100
dim = 5
optimizer = BlackBoxOptimizer(budget, dim)
func = lambda x: x**2  # Example function to optimize
optimizer(__call__, func)
optimizer.select_solution()
print(optimizer)