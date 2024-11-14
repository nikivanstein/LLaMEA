import numpy as np

class AdaptiveMutationScaleEnhancedDynamicCMAStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.sigma = 1.0
        self.mean = np.random.uniform(-5.0, 5.0, dim)
        self.C = np.identity(dim)
        self.p_sigma = np.zeros(dim)
        self.p_c = np.zeros(dim)
        self.learning_rate = 1.0

    def adaptive_mutation_scale(self, func, x, z):
        func_diff = func(self.mean) - func(x)
        if func_diff > 0:
            self.sigma *= 1.1
        else:
            self.sigma *= 0.9

    def __call__(self, func):
        for _ in range(self.budget):
            z = np.random.normal(0, 1, self.dim)
            x = self.mean + self.sigma * np.dot(self.C, z)
            self.adaptive_mutation_scale(func, x, z)
            if func(x) < func(self.mean):
                self.mean = x
            c_diff = (1 - 1/(4*self.dim)) * self.p_c + np.sqrt(1/(2*self.dim)) * z
            self.C = np.dot(np.dot(np.identity(self.dim) - np.outer(c_diff, c_diff) * 2/(np.linalg.norm(c_diff)**2), self.C), np.identity(self.dim))
            p_sigma_new = (1 - 1/(2*self.dim)) * self.p_sigma + np.sum(self.C * z) / np.linalg.norm(self.C * z)
            self.p_sigma = p_sigma_new
            self.sigma *= np.exp((np.linalg.norm(self.p_sigma) - 1) / np.sqrt(self.dim + 3) * 0.3 * self.learning_rate)
            self.mean += np.random.normal(0, 0.1, self.dim)
            self.learning_rate *= 0.999  # Dynamic learning rate adjustment

            # Differential Evolution Strategy
            u = self.mean + 0.1 * np.random.normal(0, 1, self.dim) * (self.mean - x)
            if func(u) < func(self.mean):
                self.mean = u

        return self.mean