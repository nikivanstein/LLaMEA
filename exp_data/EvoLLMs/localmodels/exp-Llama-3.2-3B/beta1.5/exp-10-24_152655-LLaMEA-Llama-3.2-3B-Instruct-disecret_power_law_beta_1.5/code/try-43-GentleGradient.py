import numpy as np

class GentleGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.step_size = 0.1

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        for _ in range(self.budget - self.f_evals):
            # Initialize a list of random candidates
            candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1))

            # Evaluate the candidates
            f_candidates = func(candidates)

            # Update the best solution
            f_evals = f_candidates[0]
            x_best = candidates[0]
            f_evals_best = f_evals

            # Update the best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = x_best
                self.f_evals_best = f_evals

            # Calculate the gradient of the function at the best candidate
            gradient = np.zeros((self.dim, 1))
            for i in range(self.dim):
                gradient[i] = (func(candidates + 1e-6 * np.ones((self.dim, 1))) - func(candidates - 1e-6 * np.ones((self.dim, 1)))) / (2 * 1e-6)

            # Update the step size based on the probability
            if np.random.rand() < 0.037037037037037035:
                self.step_size *= 0.9

            # Update the candidates using gradient descent
            for i in range(self.dim):
                candidates[i] = candidates[i] - self.step_size * gradient[i]

            # Ensure the candidates are within the bounds
            candidates = np.clip(candidates, self.bounds[:, 0], self.bounds[:, 1])

            # Evaluate the candidates
            f_candidates = func(candidates)

            # Update the best solution if necessary
            if f_evals < f_candidates[0]:
                self.f_best = f_evals
                self.x_best = candidates[0]
                self.f_evals_best = f_evals

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

gentle_gradient = GentleGradient(budget=10, dim=2)
x_opt = gentle_gradient(func)
print(x_opt)