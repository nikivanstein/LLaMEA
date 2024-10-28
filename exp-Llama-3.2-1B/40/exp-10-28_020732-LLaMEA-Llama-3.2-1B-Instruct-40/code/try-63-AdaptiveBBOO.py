# Description: Adaptive Black Box Optimization using Genetic Algorithm
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
        self.population = []

    def __call__(self, func):
        def func_wrapper(x):
            return func(x)
        self.func_evaluations.append(func_wrapper)
        return func_wrapper

    def initialize_population(self, population_size):
        self.population = [np.random.uniform(self.search_space, self.search_space + 1, self.dim) for _ in range(population_size)]

    def select_parents(self):
        fitnesses = [self.func_evaluations[i] for i in range(len(self.func_evaluations))]
        parents = np.random.choice(len(self.func_evaluations), self.budget, replace=False, p=fitnesses)
        return parents

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        for _ in range(self.dim):
            if np.random.rand() < 0.5:
                child[np.random.randint(0, self.dim)] = parent2[np.random.randint(0, self.dim)]
        return child

    def mutate(self, individual):
        if np.random.rand() < 0.4:
            index = np.random.randint(0, self.dim)
            individual[index] += np.random.uniform(-1, 1)
        return individual

    def optimize(self, func):
        # Initialize the search space
        x = self.search_space[np.random.choice(self.search_space.shape[0], self.budget)]

        # Perform the first evaluation
        func_value = func(x)
        print(f"Initial evaluation: {func_value}")

        # Initialize the best solution and best function value
        best_x = x
        best_func_value = func_value

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
            x = self.mutate(x)

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 5)

# Optimize the function with 1000 evaluations
bboo.initialize_population(100)
bboo.select_parents()
bboo.optimize(func)

# Plot the best solution and function value
plt.plot(bboo.population[:, 0], bboo.population[:, 1], label='Best solution')
plt.plot(bboo.population[:, 0], bboo.func_evaluations, label='Function value')
plt.legend()
plt.show()