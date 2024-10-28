import numpy as np

class EvolutionaryGradientAdaptation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=dim)
        self.f_best = np.inf
        self.x_best = None
        self.strategies = [np.random.uniform(-5.0, 5.0, size=dim) for _ in range(10)]
        self.strategies_probabilities = np.ones(10) / 10

    def __call__(self, func):
        for _ in range(self.budget):
            # Compute gradient of the objective function
            gradient = np.zeros(self.dim)
            h = 1e-1
            for i in range(self.dim):
                gradient[i] = (func(self.x + h * np.eye(self.dim)[i]) - func(self.x - h * np.eye(self.dim)[i])) / (2 * h)

            # Update the current solution using evolutionary strategy
            new_x = self.x + 0.5 * np.random.normal(0, 0.1, size=self.dim)
            new_x += 0.1 * gradient
            new_x = np.clip(new_x, -5.0, 5.0)

            # Adapt the evolutionary strategy
            self.strategies = [strategy + 0.1 * gradient for strategy in self.strategies]
            self.strategies_probabilities = np.random.choice(self.strategies, size=10, replace=True, p=self.strategies_probabilities)
            self.strategies_probabilities = self.strategies_probabilities / np.sum(self.strategies_probabilities)

            # Update the current solution
            self.x = self.strategies[np.random.choice(10, p=self.strategies_probabilities)]

            # Update the best solution
            f = func(self.x)
            if f < self.f_best:
                self.f_best = f
                self.x_best = self.x.copy()

            # Check for convergence
            if _ % 100 == 0 and np.all(np.abs(self.x - self.x_best) < 1e-6):
                print("Converged after {} iterations".format(_))