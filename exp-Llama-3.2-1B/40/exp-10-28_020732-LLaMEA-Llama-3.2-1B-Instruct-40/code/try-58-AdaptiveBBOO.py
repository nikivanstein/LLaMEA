# Description: AdaptiveBBOO: A novel metaheuristic algorithm for solving black box optimization problems
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
            x = x + np.random.uniform(-1, 1, self.dim)

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Define a new heuristic algorithm: AdaptiveBBOO with adaptive mutation strategy
class AdaptiveBBOO_with_adaptiveMutation(AdaptiveBBOO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = 0.01

    def mutate(self, individual):
        # Randomly decide whether to mutate the individual
        if np.random.rand() < self.mutation_rate:
            # Generate a new mutation point
            mutation_point = np.random.uniform(self.search_space.shape[0])
            # Swap the mutation point with a random point in the search space
            individual[mutation_point], individual[np.random.randint(0, self.search_space.shape[0])] = individual[np.random.randint(0, self.search_space.shape[0])], individual[mutation_point]
        return individual

# Create an instance of the AdaptiveBBOO_with_adaptiveMutation class
bboo = AdaptiveBBOO_with_adaptiveMutation(1000, 5)

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Print the updated population
print("Updated population:")
for algorithm in [AdaptiveBBOO, AdaptiveBBOO_with_adaptiveMutation]:
    print(algorithm.__class__.__name__)
    print(algorithm.__init__(1000, 5))
    print(algorithm.optimize(func))