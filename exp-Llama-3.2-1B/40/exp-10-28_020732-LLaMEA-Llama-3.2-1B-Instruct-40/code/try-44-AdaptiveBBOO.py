# Description: AdaptiveBBOO - An adaptive black box optimization algorithm using a combination of random search and evolutionary strategies.
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

        # Initialize the population of individuals
        population = [x for x in self.search_space]

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
            population = [x + np.random.uniform(-1, 1, self.dim) for x in population]

            # Select the fittest individual
            fittest_individual = population[np.argmax([func(x) for x in population])]

            # Perform crossover to create a new individual
            new_individual = fittest_individual + np.random.uniform(-1, 1, self.dim)

            # Perform mutation to introduce randomness
            new_individual = new_individual + np.random.uniform(-1, 1, self.dim)

            # Add the new individual to the population
            population.append(new_individual)

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBO()

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Plot the results
plt.plot(bboo.func_evaluations)
plt.xlabel('Evaluation Count')
plt.ylabel('Function Value')
plt.title('BBOB Test Suite')
plt.show()

An exception occured: Traceback (most recent call last):
  File "/root/LLaMEA/llamea/llamea.py", line 187, in initialize_single
    new_individual = self.evaluate_fitness(new_individual)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/LLaMEA/llamea/llamea.py", line 264, in evaluate_fitness
    updated_individual = self.f(individual, self.logger)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/LLaMEA/mutation_exp.py", line 32, in evaluateBBOB
    exec(code, globals())
  File "<string>", line 58, in <module>
NameError: name 'AdaptiveBBOO' is not defined