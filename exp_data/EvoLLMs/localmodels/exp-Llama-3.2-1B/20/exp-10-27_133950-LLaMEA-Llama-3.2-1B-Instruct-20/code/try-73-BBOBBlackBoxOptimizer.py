import numpy as np
from scipy.optimize import minimize

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0

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

    def update_strategy(self, individual, fitness):
        # Refine the strategy by changing the individual lines of the selected solution
        # This is done by changing the probability of selecting the next individual based on its fitness
        if fitness > 0.8:
            # Increase the probability of selecting the next individual
            self.update_probability(individual, 0.8)
        else:
            # Decrease the probability of selecting the next individual
            self.update_probability(individual, 0.2)

    def update_probability(self, individual, probability):
        # Update the probability of selecting the next individual
        # This is done by adding a Gaussian noise to the individual's fitness
        noise = np.random.normal(0, 1, 1)
        individual += noise
        individual /= np.abs(individual)
        individual = np.clip(individual, 0, 1)
        individual = np.exp(-(individual - 0.5) ** 2 / 2)
        individual /= np.sum(individual)
        individual = np.clip(individual, 0, 1)

        # Update the probability of selecting the next individual
        self.probability = probability / np.sum([individual ** 2 for individual in self.probability])