import numpy as np

class DynamicFireworkAlgorithm:
    def __init__(self, budget, dim, num_fireworks=20, explosion_amp=0.1, min_spark_amp=0.01, max_spark_amp=0.1):
        self.budget = budget
        self.dim = dim
        self.num_fireworks = num_fireworks
        self.explosion_amp = explosion_amp
        self.min_spark_amp = min_spark_amp
        self.max_spark_amp = max_spark_amp

    def initialize_fireworks(self):
        return np.random.uniform(-5.0, 5.0, (self.num_fireworks, self.dim))

    def explode_firework(self, firework):
        explosion_vector = np.random.uniform(-self.explosion_amp, self.explosion_amp, self.dim)
        return firework + explosion_vector

    def spark_firework(self, firework):
        spark_amp = np.random.uniform(self.min_spark_amp, self.max_spark_amp)
        return firework + spark_amp * np.random.normal(0, 1, self.dim)

    def __call__(self, func):
        fireworks = self.initialize_fireworks()
        fireworks_fitness = np.array([func(firework) for firework in fireworks])

        for _ in range(self.budget):
            for i in range(self.num_fireworks):
                new_firework = self.explode_firework(fireworks[i]) if np.random.rand() < 0.5 else self.spark_firework(fireworks[i])
                new_fitness = func(new_firework)
                
                if new_fitness < fireworks_fitness[i]:
                    fireworks[i] = new_firework
                    fireworks_fitness[i] = new_fitness

        return fireworks[np.argmin(fireworks_fitness)]