import numpy as np

class ModifiedFireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.5, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def __call__(self, func):
        def initialize_fireflies():
            return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        def move_fireflies(fireflies):
            new_fireflies = np.copy(fireflies)
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(fireflies[j]) < func(fireflies[i]):
                        beta = self.beta0 * np.exp(-self.gamma * np.linalg.norm(fireflies[j] - fireflies[i])**2)
                        new_fireflies[i] += self.alpha * (fireflies[j] - fireflies[i]) * beta
                        new_fireflies[i] = np.clip(new_fireflies[i], -5.0, 5.0)
            return new_fireflies
        
        fireflies = initialize_fireflies()
        for _ in range(self.budget):
            fireflies = move_fireflies(fireflies)
        return min(fireflies, key=lambda x: func(x))