import numpy as np

class FireflyLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.alpha = 0.2
        self.beta = 1.0
        self.initial_alpha = 1.0
        self.step_size = 0.1

    def levy_flight(self):
        sigma1 = (np.math.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2)) / (np.math.gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2)))
        sigma2 = np.power(np.math.gamma((1 + 2 * self.beta) / 2) * np.beta * np.power(2, (self.beta - 1) / 2) / np.math.gamma(1 + self.beta), 1 / self.beta)
        u = np.random.normal(0, sigma2)
        v = np.random.normal(0, 1)
        step = sigma1 * u / np.power(abs(v), 1 / self.beta)
        return step

    def move_firefly(self, current_firefly, best_firefly):
        step = self.levy_flight()
        new_position = current_firefly + step * (best_firefly - current_firefly)
        return np.clip(new_position, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        current_firefly = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_firefly = current_firefly

        for _ in range(self.budget):
            new_firefly = self.move_firefly(current_firefly, best_firefly)
            current_fitness = func(current_firefly)
            new_fitness = func(new_firefly)

            if new_fitness < current_fitness:
                current_firefly = new_firefly
                if new_fitness < func(best_firefly):
                    best_firefly = new_firefly

        return best_firefly