import numpy as np
import random

class EAS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget
        self.adaptive_strategy = self._initialize_adaptive_strategy()

    def _initialize_adaptive_strategy(self):
        # Initialize adaptive strategy based on the current budget and dimensionality
        if self.budget > 1000 and self.dim > 5:
            adaptive_strategy = "Exploration-Exploitation"
        elif self.budget > 100 and self.dim > 3:
            adaptive_strategy = "Gradual Exploration"
        else:
            adaptive_strategy = "Random Search"

        return adaptive_strategy

    def __call__(self, func, adaptive_strategy="Random Search"):
        # Initialize x to the lower bound
        x = np.array([-5.0] * self.dim)
        
        for _ in range(self.budget):
            # Evaluate the function at the current x
            y = func(x)
            
            # If we've reached the maximum number of iterations, stop exploring
            if adaptive_strategy == "Random Search":
                # Use a random strategy to refine the solution
                if random.random() < 0.2:
                    x = self.adaptive_strategy(x)
            elif adaptive_strategy == "Exploration-Exploitation":
                # Use a learning rate to adapt the search strategy
                learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
                dx = -np.dot(x - y, np.gradient(y))
                x += learning_rate * dx
            elif adaptive_strategy == "Gradual Exploration":
                # Use a gradual increase in exploration rate
                self.explore_rate = max(0.01, self.explore_rate * 1.5)

            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

# Example usage:
func = lambda x: x**2
eas = EAS(budget=1000, dim=5)
eas(x=[-10, -5, -1, 0, 1], adaptive_strategy="Gradual Exploration")