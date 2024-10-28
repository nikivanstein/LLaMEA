import numpy as np

class EvolutionaryGradientAdaptation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=dim)
        self.f_best = np.inf
        self.x_best = None
        self.adaptation_rate = 0.075
        self.adaptation_threshold = 100

    def __call__(self, func):
        for _ in range(self.budget):
            # Compute gradient of the objective function
            gradient = np.zeros(self.dim)
            h = 1e-1
            for i in range(self.dim):
                gradient[i] = (func(self.x + h * np.eye(self.dim)[i]) - func(self.x - h * np.eye(self.dim)[i])) / (2 * h)

            # Update the current solution using evolutionary strategy
            self.x += 0.5 * np.random.normal(0, 0.1, size=self.dim)

            # Update the best solution
            f = func(self.x)
            if f < self.f_best:
                self.f_best = f
                self.x_best = self.x.copy()

            # Add gradient information to the evolutionary strategy
            self.x += 0.1 * gradient

            # Adapt the evolutionary strategy based on the problem's difficulty
            if _ % self.adaptation_threshold == 0:
                # Calculate the adaptation factor
                adaptation_factor = np.random.uniform(0.9, 1.1)

                # Update the evolutionary strategy
                self.x *= adaptation_factor

                # Check for convergence
                if np.all(np.abs(self.x - self.x_best) < 1e-6):
                    print("Converged after {} iterations".format(_))

# Example usage:
def func(x):
    return np.sum(x**2)

evg_adapt = EvolutionaryGradientAdaptation(budget=1000, dim=10)
evg_adapt("func")