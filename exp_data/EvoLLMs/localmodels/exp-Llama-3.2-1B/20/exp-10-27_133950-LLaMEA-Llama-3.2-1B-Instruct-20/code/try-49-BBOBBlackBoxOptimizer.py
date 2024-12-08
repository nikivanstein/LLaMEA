import numpy as np
from scipy.optimize import minimize

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.mutation_rate = 0.1  # Initial mutation rate
        self.mutation_bound = 0.01  # Maximum mutation rate

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def evolve(self, new_individual):
        # Refine the search space based on the optimization limit
        if self.func_evaluations >= self.budget:
            # If optimization limit reached, reduce mutation rate
            self.mutation_rate *= 0.9

        # Refine the mutation strategy based on the optimization limit
        if self.func_evaluations >= self.budget * 0.5:
            # If optimization limit reached and mutation rate is high, increase mutation rate
            self.mutation_rate *= 1.1

        # Refine the search space based on the mutation strategy
        if self.mutation_rate > self.mutation_bound:
            # If mutation rate is high, reduce the search space
            self.search_space = np.linspace(-5.0, 5.0, 50)

        # Use the new individual as the next individual in the population
        new_individual = self.evaluate_fitness(new_individual)
        return new_individual

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual using the original function
        func = lambda x: x**2
        result = minimize(func, individual, method="SLSQP", bounds=[(x, x) for x in self.search_space])
        return result.fun