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

    def refine_strategy(self, new_individual):
        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.1 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.1 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.2 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.2 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.9 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.9 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.0 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.0 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.0 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.0 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.0 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.0 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.3 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.3 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.2 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.2 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.6 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.6 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.1 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.1 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.3 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.3 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.7 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.7 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.3 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.3 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.2 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.2 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.3 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.3 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.1 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.1 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.2 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.2 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.6 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.6 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.1 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.1 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.7 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.7 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.2 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.2 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.6 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.6 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.7 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.7 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.9 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.9 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.9 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.9 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.3 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.3 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.2 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.2 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.6 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.6 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.7 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.7 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.9 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.9 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.9 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.9 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.3 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.3 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.4 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.5 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.2 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.2 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.6 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.6 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.8 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.7 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.7 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 2.9 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 2.9 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 1.9 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 1.9 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x + 0.3 for x in new_individual.lines]

        # Probability of 0.2 to change the individual lines of the selected solution to refine its strategy
        if np.random.rand() < 0.2:
            new_individual.lines = [x - 0.3 for x in new_individual.lines]

    def select_solution(self):
        return np.random.choice([True, False], p=[0.2, 0.8])