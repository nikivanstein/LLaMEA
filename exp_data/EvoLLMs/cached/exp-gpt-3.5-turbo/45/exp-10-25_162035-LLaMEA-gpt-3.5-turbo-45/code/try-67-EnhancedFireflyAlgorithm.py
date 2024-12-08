import numpy as np

class EnhancedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.gamma = 0.01

    def dynamic_step_size(self, t):
        return 1 / (1 + self.gamma * t)

    def move_firefly(self, firefly, target, t):
        step = self.dynamic_step_size(t) * self.alpha * (firefly - target) + self.levy_flight()
        new_position = firefly + step
        return np.clip(new_position, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        for t in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        attractiveness_ij = self.attractiveness(self.population[i], self.population[j])
                        self.population[i] = self.move_firefly(self.population[i], self.population[j], t) * attractiveness_ij
            best_index = np.argmin([func(ind) for ind in self.population])
            for i in range(self.dim):
                self.population[best_index][i] = np.clip(self.population[best_index][i], self.lower_bound, self.upper_bound)

        return self.population[np.argmin([func(ind) for ind in self.population])]