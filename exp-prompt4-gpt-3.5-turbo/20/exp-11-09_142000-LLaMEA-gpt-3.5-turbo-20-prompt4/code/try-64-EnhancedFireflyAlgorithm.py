import numpy as np

class EnhancedFireflyAlgorithm(FireflyAlgorithm):
    def move_firefly(self, idx, alpha=0.5, beta_min=0.2):
        for i in range(self.budget):
            if func(self.population[i]) < func(self.population[idx]):
                distance = np.linalg.norm(self.population[idx] - self.population[i])
                randomization = np.random.uniform(-1, 1, self.dim)
                self.population[idx] += alpha * np.exp(-beta_min * distance) * (self.population[i] - self.population[idx]) + randomization