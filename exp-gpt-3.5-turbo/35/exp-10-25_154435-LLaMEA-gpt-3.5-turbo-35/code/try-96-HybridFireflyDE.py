import numpy as np

class HybridFireflyDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.firefly_pop_size = 20
        self.de_pop_size = 10
        self.lb = -5.0
        self.ub = 5.0
        self.alpha = 0.2
        self.beta_min = 0.2
        self.gamma = 0.1

    def __call__(self, func):
        def initialize_firefly_pop(pop_size, dim, lb, ub):
            return np.random.uniform(lb, ub, (pop_size, dim))

        def levy_flight(dim):
            sigma = (np.math.gamma(1 + self.beta_min) * np.sin(np.pi * self.beta_min / 2) / (np.math.gamma((1 + self.beta_min) / 2) * self.beta_min * 2 ** ((self.beta_min - 1) / 2))) ** (1 / self.beta_min)
            u = np.random.normal(0, sigma, dim)
            v = np.random.normal(0, 1, dim)
            step = u / (abs(v) ** (1 / self.beta_min))
            return step

        fireflies = initialize_firefly_pop(self.firefly_pop_size, self.dim, self.lb, self.ub)
        best_firefly = fireflies[np.argmin([func(f) for f in fireflies])]
        for _ in range(self.budget // self.firefly_pop_size):
            for i in range(len(fireflies)):
                for j in range(self.de_pop_size):
                    candidate = fireflies[i] + self.alpha * levy_flight(self.dim) * (fireflies[np.random.randint(self.firefly_pop_size)] - fireflies[np.random.randint(self.firefly_pop_size)])
                    if func(candidate) < func(fireflies[i]):
                        fireflies[i] = candidate
            new_best = fireflies[np.argmin([func(f) for f in fireflies])]
            if func(new_best) < func(best_firefly):
                best_firefly = new_best
        return best_firefly