# Description: AdaptiveBBOO is a novel metaheuristic algorithm for solving black box optimization problems.
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

# Define a mutation strategy to refine the selected solution
def mutate(individual):
    if np.random.rand() < 0.4:
        individual = np.clip(individual, -5.0, 5.0)
    return individual

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBO()

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(bboo.func_evaluations[::100], label='Evaluation')
plt.plot(bboo.func_evaluations[100::100], label='Optimal Solution')
plt.plot(bboo.func_evaluations[200::100], label='Refined Optimal Solution')
plt.xlabel('Evaluation')
plt.ylabel('Function Value')
plt.title('AdaptiveBBOO Results')
plt.legend()
plt.show()

# An exception occured: Traceback (most recent call last):
#   File "/root/LLaMEA/llamea/llamea.py", line 187, in initialize_single
#     new_individual = self.evaluate_fitness(new_individual)
#   File "/root/LLaMEA/mutation_exp.py", line 32, in evaluateBBOB
#     exec(code, globals())
#   File "<string>", line 58, in <module>
#     NameError: name 'AdaptiveBBOO' is not defined