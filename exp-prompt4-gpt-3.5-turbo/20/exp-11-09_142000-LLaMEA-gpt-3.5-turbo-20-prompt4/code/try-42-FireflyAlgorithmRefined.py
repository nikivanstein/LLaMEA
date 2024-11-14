import numpy as np

class FireflyAlgorithmRefined(FireflyAlgorithm):
    def levy_flight(self, step_size=0.1):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        step = step_size * u / abs(v)**(1 / beta)
        return step

    def move_firefly(self, idx, alpha=0.5, beta_min=0.2):
        for i in range(self.budget):
            if func(self.population[i]) < func(self.population[idx]):
                distance = np.linalg.norm(self.population[idx] - self.population[i])
                step = self.levy_flight()
                self.population[idx] += alpha * step * np.exp(-beta_min * distance) * (self.population[i] - self.population[idx])