import numpy as np

class ImprovedBatAlgorithm(BatAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.pulse_rate = 0.8

    def __call__(self, func):
        for _ in range(self.budget):
            frequencies = self.f_min + (self.f_max - self.f_min) * np.random.rand(self.population_size)
            self.v += (self.population - func(self.population)) * frequencies[:, None]
            self.population = np.clip(self.population + self.v, self.v_min, self.v_max)
            for i in range(self.population_size):
                if np.random.rand() < self.pulse_rate:
                    self.population[i] = np.clip(self.population[i] + 0.01 * np.random.randn(self.dim), self.v_min, self.v_max)
                else:
                    if np.random.rand() > self.Q[i]:
                        self.population[i] = func(np.random.uniform(-5.0, 5.0, self.dim))
            self.Q = self.Q_min + (self.Q_max - self.Q_min) * np.random.rand(self.population_size)
            self.A_min = self.alpha * self.A_min
            self.A = self.A_min + (self.A_max - self.A_min) * np.random.rand(self.population_size)
            self.f_min = self.f_min + self.gamma
            self.f_max = self.f_max * self.alpha
        return self.population