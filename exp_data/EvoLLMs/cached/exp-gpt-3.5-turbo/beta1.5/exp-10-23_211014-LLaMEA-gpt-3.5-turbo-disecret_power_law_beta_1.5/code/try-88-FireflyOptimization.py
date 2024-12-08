import numpy as np

class FireflyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def attractiveness(self, light_intensity, distance):
        return light_intensity / (1 + distance)
    
    def move_firefly(self, source, target, attractiveness, beta):
        return source + attractiveness * np.exp(-beta * np.linalg.norm(target - source)) * (target - source) + 0.01 * np.random.randn(self.dim)
    
    def optimize(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        attractiveness_ij = self.attractiveness(1, np.linalg.norm(self.population[i] - self.population[j]))
                        self.population[i] = self.move_firefly(self.population[i], self.population[j], attractiveness_ij, 1.0)
        return self.population[np.argmin([func(x) for x in self.population])]