import numpy as np

class ImprovedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.attractiveness_scaling = 1.0

    def __call__(self, func):
        def move_fireflies(alpha=1.0, beta=1.0, gamma=0.5):
            new_population = np.copy(self.population)
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        attractiveness_value = self.attractiveness(i, j)
                        new_population[i] += alpha * attractiveness_value * (self.population[j] - self.population[i]) + gamma * np.random.uniform(-1, 1, self.dim)
            self.population = new_population

        self.population = self.initialize_population()
        for _ in range(self.budget):
            move_fireflies()
            self.attractiveness_scaling = 1.0 - (_ / self.budget)  # Adaptive attractiveness scaling
        best_solution = min(self.population, key=lambda x: func(x))
        return best_solution