import numpy as np

class ImprovedBatAlgorithm(BatAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.epsilon = 0.01

    def __call__(self, func):
        for _ in range(self.budget):
            frequencies = np.clip(self.f_min + (self.f_max - self.f_min) * np.random.rand(self.population_size), 0, 1)
            self.v += (self.population - func(self.population)) * frequencies[:, None]
            self.population = np.clip(self.population + self.v, self.v_min, self.v_max)
            for i in range(self.population_size):
                if np.random.rand() > self.Q[i]:
                    self.population[i] = func(np.random.uniform(-5.0, 5.0, self.dim))
            self.Q = np.clip(self.Q_min + (self.Q_max - self.Q_min) * np.random.rand(self.population_size), 0, 2)
            self.A_min = self.alpha * self.A_min
            self.A = np.clip(self.A_min + (self.A_max - self.A_min) * np.random.rand(self.population_size), 0, 2)
            self.f_min = np.clip(self.f_min + self.gamma, 0, 1)
            self.f_max = np.clip(self.f_max * self.alpha, 0, 1)
        return self.population