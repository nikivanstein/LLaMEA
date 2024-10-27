import numpy as np
import random

class MultiDirectionalHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.p = 0.5  # probability of accepting worse solutions
        self.h = 10  # harmony size
        self.d = 5  # number of directions
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.h, self.dim))
        self.best_x = np.copy(self.x[0])
        self.best_f = func(self.x[0])
        self.p_line_search = 0.15  # probability of changing the individual lines

    def __call__(self, func):
        for _ in range(self.budget):
            # Generate new harmonies
            new_x = np.copy(self.x)
            for i in range(self.h):
                for j in range(self.d):
                    # Randomly select a direction
                    dx = np.random.uniform(-self.upper_bound, self.upper_bound)
                    new_x[i] += dx * np.random.uniform(0, 1)
                    new_x[i] = np.clip(new_x[i], self.lower_bound, self.upper_bound)

            # Calculate the objective function values
            f_new = [func(xi) for xi in new_x]

            # Update the best solution
            if np.min(f_new) < self.best_f:
                self.best_x = new_x[np.argmin(f_new)]
                self.best_f = np.min(f_new)

            # Acceptance criteria
            if np.random.rand() < self.p or np.min(f_new) < self.best_f:
                # Probabilistic line search to refine the strategy
                if np.random.rand() < self.p_line_search:
                    # Generate new individuals by changing one line at a time
                    new_individual = self.x.copy()
                    for k in range(self.dim):
                        new_individual[k] += np.random.uniform(-0.1, 0.1)
                        new_individual[k] = np.clip(new_individual[k], self.lower_bound, self.upper_bound)
                    new_f = func(new_individual)
                    if new_f < self.best_f:
                        self.best_x = new_individual
                        self.best_f = new_f

                self.x = new_x
                self.best_f = np.min(f_new)

            # Print the best solution
            print(f"Best solution: f({self.best_x}) = {self.best_f}")

# Example usage
if __name__ == "__main__":
    budget = 100
    dim = 10
    multi_directional_harmony_search = MultiDirectionalHarmonySearch(budget, dim)
    multi_directional_harmony_search("func")