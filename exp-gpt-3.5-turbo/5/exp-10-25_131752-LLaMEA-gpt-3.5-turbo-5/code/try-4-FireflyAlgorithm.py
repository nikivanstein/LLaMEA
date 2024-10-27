import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def __call__(self, func):
        def attractiveness(distance):
            return self.beta0 * np.exp(-self.gamma * distance**2)

        def move_fireflies(fireflies):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(fireflies[i]) > func(fireflies[j]):
                        distance = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta = attractiveness(distance)
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + self.alpha * np.random.uniform(-1, 1, self.dim)
                        fireflies[i] = np.clip(fireflies[i], -5.0, 5.0)
            return fireflies

        fireflies = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            fireflies = move_fireflies(fireflies)
        return fireflies[np.argmin([func(f) for f in fireflies])]