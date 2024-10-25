import numpy as np
import random

class EvolutionaryGradientAdapt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=dim)
        self.f_best = np.inf
        self.x_best = None
        self.step_size = 0.1
        self.learning_rate = 0.1
        self.convergence_threshold = 1e-6
        self.convergence_count = 0

    def __call__(self, func):
        for _ in range(self.budget):
            # Compute gradient of the objective function
            gradient = np.zeros(self.dim)
            h = 1e-1
            for i in range(self.dim):
                gradient[i] = (func(self.x + h * np.eye(self.dim)[i]) - func(self.x - h * np.eye(self.dim)[i])) / (2 * h)

            # Update the current solution using evolutionary strategy
            self.x += self.step_size * np.random.normal(0, 0.1, size=self.dim)

            # Update the best solution
            f = func(self.x)
            if f < self.f_best:
                self.f_best = f
                self.x_best = self.x.copy()

            # Add gradient information to the evolutionary strategy
            self.x += self.learning_rate * gradient

            # Adapt the step size and learning rate dynamically
            if _ % 100 == 0:
                if np.all(np.abs(self.x - self.x_best) < self.convergence_threshold):
                    self.convergence_count += 1
                else:
                    self.convergence_count = 0
                if self.convergence_count >= 10:
                    self.step_size *= 0.9
                    self.learning_rate *= 0.9
                elif self.convergence_count < 5:
                    self.step_size *= 1.1
                    self.learning_rate *= 1.1

            # Check for convergence
            if _ % 100 == 0 and np.all(np.abs(self.x - self.x_best) < self.convergence_threshold):
                print("Converged after {} iterations".format(_))