import numpy as np
from scipy.optimize import minimize

class AdaptiveEvolutionaryOptimizer:
    def __init__(self, budget, dim, adaptive_threshold=0.2):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.adaptive_threshold = adaptive_threshold
        self.learning_rate = 0.01
        self.adaptive_gain = 0.5

    def __call__(self, func, adaptive=True):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            if adaptive:
                self.update_adaptive_strategy(x)
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def update_adaptive_strategy(self, x):
        if np.random.rand() < self.adaptive_threshold:
            # Refine the strategy by changing the learning rate
            self.learning_rate *= self.adaptive_gain
            print(f"Learning rate updated to {self.learning_rate}")