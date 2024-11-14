import numpy as np

class EnhancedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        
    def levy_flight(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (abs(v) ** (1 / beta))
        return step

    def move_fireflies(self, alpha=1.0, beta=1.0, gamma=0.5):
        new_population = np.copy(self.population)
        for i in range(self.budget):
            for j in range(self.budget):
                if func(self.population[j]) < func(self.population[i]):
                    step = alpha * np.exp(-beta * np.linalg.norm(self.population[j] - self.population[i]) ** 2) * (self.population[j] - self.population[i]) + gamma * self.levy_flight()
                    new_population[i] += step
        self.population = new_population

enhanced_algorithm = EnhancedFireflyAlgorithm(budget, dim)
best_solution = enhanced_algorithm(func)