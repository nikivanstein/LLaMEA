class ImprovedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def move_firefly(self, idx, alpha=0.5, beta_min=0.2):
        alpha = 1.0 - ((self.budget - idx) / self.budget)  # Dynamic alpha based on iteration count
        for i in range(self.budget):
            if func(self.population[i]) < func(self.population[idx]):
                distance = np.linalg.norm(self.population[idx] - self.population[i])
                self.population[idx] += alpha * np.exp(-beta_min * distance) * (self.population[i] - self.population[idx])