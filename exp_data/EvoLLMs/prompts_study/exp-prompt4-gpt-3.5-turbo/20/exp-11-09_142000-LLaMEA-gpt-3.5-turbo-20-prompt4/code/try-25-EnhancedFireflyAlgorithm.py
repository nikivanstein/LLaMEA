import numpy as np

class EnhancedFireflyAlgorithm(FireflyAlgorithm):
    def levy_flight(self, scale=0.1):
        return np.random.standard_cauchy(size=self.dim) * scale / (np.abs(np.random.normal()) ** (1 / self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        self.move_firefly(i)
                        self.population[i] += self.levy_flight()
        best_idx = np.argmin([func(ind) for ind in self.population])
        return self.population[best_idx]