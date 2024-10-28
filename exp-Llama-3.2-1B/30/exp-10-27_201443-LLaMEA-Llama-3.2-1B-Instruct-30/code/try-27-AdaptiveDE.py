import numpy as np
import random

class AdaptiveDE:
    def __init__(self, budget, dim, alpha=0.5, beta=0.5, sigma=0.1, mu=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.mu = mu
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population_size = 100
        self.population = [np.random.uniform(self.search_space, self.search_space) for _ in range(self.population_size)]

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Adaptive scaling
            self.mu = self.alpha * self.mu + (1 - self.alpha) * self.search_space[0]
            self.mu = np.clip(self.mu, -5.0, 5.0)

            # Local search
            for _ in range(self.population_size):
                x = self.population[random.randint(0, self.population_size - 1)]
                f_x = func(x)
                if np.abs(f_x) < 1e-6:  # stop if the function value is close to zero
                    break
            self.population.append(x)

            # Differential evolution
            self.func_evaluations += 1
            func_value = func(self.population[-1])
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
        return func(self.population[-1])

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

adaptive_de = AdaptiveDE(1000, 2)  # 1000 function evaluations, 2 dimensions
print(adaptive_de(test_function))  # prints a random value between -10 and 10