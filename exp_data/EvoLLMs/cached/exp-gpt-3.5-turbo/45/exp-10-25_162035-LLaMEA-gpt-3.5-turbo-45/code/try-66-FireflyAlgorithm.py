import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.alpha = 0.2
        self.beta0 = 1.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

    def levy_flight(self):
        beta = self.beta0 / np.sqrt(self.dim)
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        levy = u / (np.abs(v) ** (1 / beta))
        return levy

    def attractiveness(self, x, y):
        return np.exp(-np.linalg.norm(x - y))

    def move_firefly(self, firefly, target):
        step = self.alpha * (firefly - target) + self.levy_flight()
        new_position = firefly + step
        return np.clip(new_position, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        for t in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        attractiveness_ij = self.attractiveness(self.population[i], self.population[j])
                        self.population[i] = self.move_firefly(self.population[i], self.population[j]) * attractiveness_ij
            best_index = np.argmin([func(ind) for ind in self.population])
            for i in range(self.dim):
                self.population[best_index][i] = np.clip(self.population[best_index][i], self.lower_bound, self.upper_bound)

        return self.population[np.argmin([func(ind) for ind in self.population])]