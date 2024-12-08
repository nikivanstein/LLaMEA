import numpy as np

class EnhancedBatAlgorithm(BatAlgorithm):
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.1):
        super().__init__(budget, dim, population_size, loudness, pulse_rate, alpha, gamma)

    def __call__(self, func):
        while self.budget > 0:
            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    new_solution = self.population[i] + self.alpha * (np.random.rand(self.dim) - 0.5)
                else:
                    new_solution = self.best_solution + self.gamma * np.random.normal(0, 1, self.dim)

                new_solution = np.clip(new_solution, -5.0, 5.0)
                new_fitness = func(new_solution)
                
                if (np.random.rand() < self.loudness) and (new_fitness <= self.fitness[i]):
                    self.population[i] = new_solution
                    self.fitness[i] = new_fitness

                    if new_fitness < self.best_fitness:
                        self.best_solution = new_solution
                        self.best_fitness = new_fitness

            self.loudness *= np.exp(-0.005 * self.budget)
            self.pulse_rate = 0.2 + 0.8 * (1 - np.exp(-0.05 * self.budget))

            self.budget -= self.population_size

        return self.best_solution