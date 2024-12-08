import numpy as np
import random

class PSM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.pbest = np.zeros((budget, dim))
        self.rbest = np.zeros((budget, dim))
        self.memory = {}
        self.probability = 0.35

    def __call__(self, func):
        for i in range(self.budget):
            # Evaluate the function at the current particles
            values = func(self.particles[i])
            # Update the personal best and global best
            self.pbest[i] = self.particles[i]
            self.rbest[i] = values
            # Check if the current best is better than the stored best
            if np.any(values < self.rbest, axis=1):
                self.memory[i] = values
                self.rbest[np.argmin(self.rbest, axis=1)] = values
            # Update the particles using the PSO update rule
            self.particles[i] = self.particles[i] + 0.5 * (self.pbest[i] - self.particles[i]) + 0.5 * (self.rbest - self.particles[i])
        # Return the best solution found
        return self.rbest[np.argmin(self.rbest, axis=1)]

    def refine_solution(self):
        # Refine the solution by changing individual lines with a probability of 0.35
        for i in range(self.budget):
            if random.random() < self.probability:
                # Randomly select a dimension to change
                dim = random.randint(0, self.dim-1)
                # Randomly select a new value between -5.0 and 5.0
                new_value = np.random.uniform(-5.0, 5.0)
                # Change the value of the selected dimension
                self.particles[i, dim] = new_value
                # Update the personal best and global best
                self.pbest[i, dim] = self.particles[i]
                self.rbest[i, dim] = func(self.particles[i])
                # Check if the current best is better than the stored best
                if np.any(self.rbest < self.memory[i], axis=1):
                    self.memory[i] = self.rbest
        # Return the refined solution
        return self.rbest[np.argmin(self.rbest, axis=1)]

# Example usage
def func(x):
    return np.sum(x**2)

psm = PSM(budget=10, dim=2)
best_solution = psm(func)
print(best_solution)

refined_solution = psm.refine_solution()
print(refined_solution)