import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.5, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)

    def move_firefly(self, firefly, best_firefly, func):
        r = np.linalg.norm(firefly - best_firefly)
        beta = self.attractiveness(r)
        epsilon = self.alpha * (np.random.rand(self.dim) - 0.5)
        return firefly + beta * (best_firefly - firefly) + epsilon

    def __call__(self, func):
        fireflies = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        intensities = np.array([func(firefly) for firefly in fireflies])
        best_index = np.argmin(intensities)
        best_firefly = fireflies[best_index]

        for _ in range(self.budget):
            for i in range(self.budget):
                if intensities[i] < intensities[best_index]:
                    best_index = i
                    best_firefly = fireflies[i]
            for i in range(self.budget):
                fireflies[i] = self.move_firefly(fireflies[i], best_firefly, func)
                intensities[i] = func(fireflies[i])

        return best_firefly