# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget
        self.exploration_strategy = self._initialize_exploration_strategy()

    def _initialize_exploration_strategy(self):
        if self.dim == 1:
            return lambda x: np.random.uniform(-5.0, 5.0)
        elif self.dim == 2:
            return lambda x: np.random.uniform(-5.0, 5.0, (self.dim,)).flatten()
        else:
            raise ValueError("Only 1D and 2D optimization is supported")

    def __call__(self, func):
        def inner(x):
            return func(x)

        # Initialize x to the lower bound
        x = np.array([-5.0] * self.dim)

        for _ in range(self.budget):
            # Evaluate the function at the current x
            y = inner(x)

            # If we've reached the maximum number of iterations, stop exploring
            if self.explore_count >= self.max_explore_count:
                break

            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break

            # Learn a new direction using gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(x - inner(x), self.exploration_strategy(x))
            x += learning_rate * dx

            # Update the exploration count
            self.explore_count += 1

            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break

        return x

    def evaluate_fitness(self, individual, logger):
        new_individual = individual
        fitness = self.f(new_individual, logger)
        logger.update_fitness(individual, fitness)
        return new_individual

    def f(self, individual, logger):
        # Evaluate the function at the individual using the BBOB test suite
        func = self._get_function(individual)
        return func(individual)

    def _get_function(self, individual):
        # Select a noiseless function from the BBOB test suite
        functions = ["c_1", "c_2", "c_3", "c_4", "c_5", "c_6", "c_7", "c_8", "c_9", "c_10", "c_11", "c_12", "c_13", "c_14", "c_15"]
        func = random.choice(functions)
        return eval("lambda x: " + func)

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
mgdalr = MGDALR(budget=100, dim=2)
mgdalr.explore_strategy = mgdalr.exploration_strategy(inner)
mgdalr.optimize(func="c_1", logger=logging.getLogger())