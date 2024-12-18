class ImprovedDynamicBatAlgorithm(DynamicBatAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.loudness_min = 0.2
        self.loudness_max = 1.0

    def __call__(self, func):
        for _ in range(self.budget):
            frequencies = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand(self.population_size)
            loudness_values = self.loudness_min + (self.loudness_max - self.loudness_min) * np.exp(-0.1 * np.arange(self.population_size))
            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    self.velocities[i] += (self.population[i] - self.best_solution) * frequencies[i] * loudness_values[i]
                self.population[i] += self.velocities[i]
                fitness = func(self.population[i])
                if fitness < self.best_fitness:
                    self.best_solution = self.population[i]
                    self.best_fitness = fitness
        return self.best_solution