import numpy as np
import random

class AdaptiveMSGD:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.mutation_rate = 0.01
        self.population_history = []

    def __call__(self, func):
        # Evaluate the function a specified number of times
        num_evals = min(self.budget, self.dim)
        func_values = [func(i / self.budget) for i in range(num_evals)]

        # Initialize the current best solution
        best_solution = None
        best_score = -np.inf

        # Iterate over the population
        for _ in range(self.population_size):
            # Randomly select a solution from the current population
            solution = random.uniform(self.search_space)

            # Evaluate the function at the solution
            func_value = func(solution)

            # If this solution is better than the current best, update the best solution
            if func_value > best_score:
                best_solution = solution
                best_score = func_value

            # Add the solution to the population history
            self.population_history.append((solution, func_value))

            # If the population is full, apply adaptive mutation
            if len(self.population) >= self.population_size:
                # Select the fittest solutions to adapt
                fittest_solutions = sorted(self.population, key=lambda x: x[1], reverse=True)[:self.population_size // 2]

                # Apply adaptive mutation
                for fittest_solution in fittest_solutions:
                    # Randomly select a mutation direction
                    mutation_direction = np.random.uniform(-1, 1, self.dim)

                    # Apply mutation
                    mutation_solution = fittest_solution[0] + mutation_direction

                    # Ensure the mutation solution is within the search space
                    mutation_solution = np.clip(mutation_solution, self.search_space[0], self.search_space[1])

                    # Evaluate the function at the mutated solution
                    func_value = func(mutation_solution)

                    # If this solution is better than the current best, update the best solution
                    if func_value > best_score:
                        best_solution = mutation_solution
                        best_score = func_value

        # Return the best solution found
        return best_solution

    def run(self, func):
        # Run the function with the current population
        func_values = [func(i / self.budget) for i in range(self.budget)]
        best_solution = max(self.population, key=lambda x: x[1])

        # Return the best solution found
        return best_solution