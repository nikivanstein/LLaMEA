import numpy as np

class ImprovedFireflyAlgorithm(FireflyAlgorithm):
    def levy_flight(self, alpha=1.5, beta=0.5):
        sigma = ((np.math.gamma(1 + beta) * np.math.sin(np.pi * beta / 2)) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / abs(v) ** (1 / beta)
        return alpha * step

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        self.move_firefly(i)
                        # Integrate Levy flights for exploration
                        levy_step = self.levy_flight()
                        self.population[i] += levy_step
        best_idx = np.argmin([func(ind) for ind in self.population])
        return self.population[best_idx]