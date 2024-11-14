import numpy as np

class ImprovedBatAlgorithm(BatAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.Q = np.random.uniform(self.Q_min, self.Q_max, self.population_size)
        self.A = np.random.uniform(self.A_min, self.A_max, self.population_size)

    def __call__(self, func):
        for _ in range(self.budget):
            frequencies = self.f_min + (self.f_max - self.f_min) * np.random.rand(self.population_size)
            self.v += (self.population - func(self.population)) * frequencies[:, None]
            self.population = np.clip(self.population + self.v, self.v_min, self.v_max)
            for i in range(self.population_size):
                if np.random.rand() > self.Q[i]:
                    temp = func(np.random.uniform(-5.0, 5.0, self.dim))
                    if temp < func(self.population[i]):
                        self.population[i] = temp
            self.Q = self.Q_min + (self.Q_max - self.Q_min) * np.random.rand(self.population_size)
            self.A = self.alpha * self.A
            self.f_min += self.gamma
            self.f_max *= self.alpha
        return self.population