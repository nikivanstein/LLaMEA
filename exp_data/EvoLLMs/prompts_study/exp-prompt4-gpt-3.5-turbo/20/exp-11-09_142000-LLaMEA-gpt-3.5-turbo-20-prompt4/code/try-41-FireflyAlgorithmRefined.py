import numpy as np

class FireflyAlgorithmRefined(FireflyAlgorithm):
    def levy_flight(self, scale=0.5):
        sigma = (np.math.gamma(1 + scale) * np.math.sin(np.pi * scale / 2) / (np.math.gamma((1 + scale) / 2) * scale * 2 ** ((scale - 1) / 2))) ** (1 / scale)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        return u / (np.abs(v) ** (1 / scale))

    def move_firefly(self, idx, alpha=0.5, beta_min=0.2):
        for i in range(self.budget):
            if func(self.population[i]) < func(self.population[idx]):
                distance = np.linalg.norm(self.population[idx] - self.population[i])
                step = np.exp(-beta_min * distance) * (self.population[i] - self.population[idx])
                step += self.levy_flight()  # Incorporate Levy flight step
                self.population[idx] += alpha * step

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        self.move_firefly(i)
        best_idx = np.argmin([func(ind) for ind in self.population])
        return self.population[best_idx]