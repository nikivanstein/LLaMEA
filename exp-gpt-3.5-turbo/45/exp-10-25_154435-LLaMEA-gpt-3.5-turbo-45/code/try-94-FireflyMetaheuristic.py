# import numpy as np

class FireflyMetaheuristic:
    def __init__(self, budget, dim, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)

    def distance(self, a, b):
        return np.linalg.norm(a - b)

    def move_fireflies(self, fireflies, func):
        for i in range(len(fireflies)):
            for j in range(len(fireflies)):
                if func(fireflies[j]) < func(fireflies[i]):
                    r = self.distance(fireflies[i], fireflies[j])
                    beta = self.attractiveness(r)
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) + self.alpha * np.random.normal(0, 1, self.dim)
        return fireflies

    def __call__(self, func):
        fireflies = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            fireflies = self.move_fireflies(fireflies, func)
        best_solution = fireflies[np.argmin([func(firefly) for firefly in fireflies])]
        return best_solution