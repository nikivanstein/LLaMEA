import numpy as np

class FireflyMetaheuristic:
    def __init__(self, budget, dim, alpha=0.2, beta_0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta_0 = beta_0
        self.gamma = gamma

    def attractiveness(self, x_i, x_j):
        return self.beta_0 * np.exp(-self.gamma * np.linalg.norm(x_i - x_j))

    def move_firefly(self, x_i, x_j, attractiveness):
        return x_i + self.alpha * attractiveness * (x_j - x_i) + np.random.normal(0, 1, self.dim)

    def __call__(self, func):
        fireflies = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(fireflies[i]) > func(fireflies[j]):
                        attractiveness_ij = self.attractiveness(fireflies[i], fireflies[j])
                        fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], attractiveness_ij)
        best_solution = fireflies[np.argmin([func(firefly) for firefly in fireflies])]
        return best_solution