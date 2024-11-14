import numpy as np

class ImprovedBatAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.f_min = 0.0
        self.f_max = 1.0
        self.alpha = 0.9
        self.gamma = 0.5
        self.A_min = 0.0
        self.A_max = 2.0
        self.Q_min = 0.0
        self.Q_max = 2.0
        self.Q = np.random.uniform(self.Q_min, self.Q_max, self.population_size)
        self.v = np.zeros((self.population_size, self.dim))
        self.v_min = -5.0
        self.v_max = 5.0
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            frequencies = self.f_min + (self.f_max - self.f_min) * np.random.rand(self.population_size)
            self.v += (self.population - func(self.population)) * frequencies[:, None]
            self.population = np.clip(self.population + self.v, self.v_min, self.v_max)
            for i in range(self.population_size):
                if np.random.rand() > self.Q[i]:
                    self.population[i] = func(np.random.uniform(-5.0, 5.0, self.dim))
                if np.random.rand() < 0.1:  # Modified part for enhanced diversity
                    self.population[i] = np.random.uniform(-5.0, 5.0, self.dim)
            self.Q = self.Q_min + (self.Q_max - self.Q_min) * np.random.rand(self.population_size)
            self.A_min = self.alpha * self.A_min
            self.A = self.A_min + (self.A_max - self.A_min) * np.random.rand(self.population_size)
            self.f_min = self.f_min + self.gamma
            self.f_max = self.f_max * self.alpha
        return self.population