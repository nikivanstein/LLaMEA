class DynamicBatAlgorithmImproved(DynamicBatAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.adaptive_pulse_rate = 0.5

    def __call__(self, func):
        for _ in range(self.budget):
            frequencies = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand(self.population_size)
            for i in range(self.population_size):
                if np.random.rand() > self.adaptive_pulse_rate:
                    self.velocities[i] += (self.population[i] - self.best_solution) * frequencies[i]
                self.population[i] += self.velocities[i]
                fitness = func(self.population[i])
                if fitness < self.best_fitness:
                    self.best_solution = self.population[i]
                    self.best_fitness = fitness
                    improvement = self.best_fitness - fitness
                    self.adaptive_pulse_rate = self.pulse_rate + 0.1 * improvement if improvement > 0 else self.pulse_rate
        return self.best_solution