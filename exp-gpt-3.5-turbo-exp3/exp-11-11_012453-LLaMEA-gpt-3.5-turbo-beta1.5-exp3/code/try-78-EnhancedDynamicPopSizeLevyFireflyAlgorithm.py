import numpy as np

class EnhancedDynamicPopSizeLevyFireflyAlgorithm(DynamicPopSizeLevyFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = 0.2  # Attraction coefficient
        self.gamma = 0.1  # Absorption coefficient

    def update_population_size(self):
        # Update population size based on the best individual's fitness
        best_fitness = self.eval(self.population[0])
        self.pop_size = max(5, int(np.floor(10 + 5 * np.exp(-self.gamma * best_fitness)))

    def move_fireflies(self):
        # Move fireflies based on levy flight and population size adaptation
        for i in range(self.pop_size):
            for j in range(self.pop_size):
                if self.eval(self.population[i]) < self.eval(self.population[j]):
                    r = np.linalg.norm(self.population[i] - self.population[j])
                    beta = self.alpha * np.exp(-self.gamma * r**2)
                    step = self.levy_flight()
                    new_pos = self.population[i] + beta * step
                    new_pos = np.clip(new_pos, self.lb, self.ub)
                    if self.eval(new_pos) < self.eval(self.population[i]):
                        self.population[i] = new_pos