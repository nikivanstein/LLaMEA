# Description: Adaptive Black Box Optimization using Evolutionary Strategies
# Code: 
# ```python
import numpy as np
import matplotlib.pyplot as plt

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = []

    def __call__(self, func):
        def func_wrapper(x):
            return func(x)
        self.func_evaluations.append(func_wrapper)
        return func_wrapper

    def optimize(self, func):
        # Initialize the search space
        x = self.search_space[np.random.choice(self.search_space.shape[0], self.budget)]

        # Perform the first evaluation
        func_value = func(x)
        print(f"Initial evaluation: {func_value}")

        # Initialize the best solution and best function value
        best_x = x
        best_func_value = func_value

        # Initialize the population of solutions
        solutions = [x.copy() for x in self.func_evaluations]

        # Define the mutation operator
        def mutate(solution):
            if np.random.rand() < 0.4:
                return solution + np.random.uniform(-1, 1, self.dim)
            else:
                return solution

        # Define the selection operator
        def select(parent1, parent2):
            return np.random.choice([parent1, parent2], p=[0.6, 0.4])

        # Define the crossover operator
        def crossover(parent1, parent2):
            if np.random.rand() < 0.4:
                child = parent1.copy()
                child[:self.dim] = parent2[:self.dim]
                return child
            else:
                child = np.concatenate((parent1, parent2))
                return child

        # Define the selection operator
        def select_solutions(solutions):
            return select(solutions[0], solutions[1])

        # Perform the metaheuristic search
        for _ in range(self.budget):
            # Evaluate the function at the current search point
            func_value = func(x)

            # If the function value is better than the best found so far, update the best solution
            if func_value > best_func_value:
                best_x = x
                best_func_value = func_value

            # If the search space is exhausted, stop the algorithm
            if np.all(x >= self.search_space):
                break

            # Randomly perturb the search point
            x = mutate(x)

            # Select the best solution
            solutions = select_solutions(solutions)

            # Perform crossover
            if len(solutions) > 2:
                parent1, parent2 = solutions[np.random.choice(2, size=2), :]
                child = crossover(parent1, parent2)
                solutions = select_solutions(solutions)

            # Evaluate the best solution at the final search point
            func_value = func(best_x)

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)

# Optimize the function
best_x, best_func_value = bboo.optimize(func)
print(f"Best solution: {best_x}, Best function value: {best_func_value}")