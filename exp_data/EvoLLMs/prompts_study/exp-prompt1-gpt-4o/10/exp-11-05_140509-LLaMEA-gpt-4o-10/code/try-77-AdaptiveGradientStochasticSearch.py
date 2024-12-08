import numpy as np

class AdaptiveGradientStochasticSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.learning_rate = 0.1
        self.sigma = 0.1

    def __call__(self, func):
        x = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_x = x.copy()
        best_f = func(best_x)
        self.evaluations += 1

        while self.evaluations < self.budget:
            perturbation = np.random.normal(0, self.sigma, self.dim)
            grad_approx = (func(x + perturbation) - func(x - perturbation)) / (2 * self.sigma)
            self.evaluations += 2

            x = x - self.learning_rate * grad_approx + perturbation
            x = np.clip(x, self.lower_bound, self.upper_bound)

            current_f = func(x)
            self.evaluations += 1

            if current_f < best_f:
                best_f = current_f
                best_x = x.copy()
                self.learning_rate *= 1.2  # Adaptive increase
                self.sigma *= 0.9  # Reduce perturbation to focus on exploitation
            else:
                self.learning_rate *= 0.8  # Adaptive decrease
                self.sigma *= 1.1  # Increase perturbation for exploration

        return best_x