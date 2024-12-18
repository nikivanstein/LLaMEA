class EnhancedDynamicBatAlgorithm(DynamicBatAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.adaptive_mutation_factor = 0.1
        self.prev_improvement_rate = 0.0

    def __call__(self, func):
        improvement_rates = []
        for _ in range(self.budget):
            frequencies = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand(self.population_size)
            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    self.velocities[i] += (self.population[i] - self.best_solution) * frequencies[i] * self.adaptive_mutation_factor
                self.population[i] += self.velocities[i]
                fitness = func(self.population[i])
                improvement_rate = (self.best_fitness - fitness) / self.best_fitness
                improvement_rates.append(improvement_rate)
                if fitness < self.best_fitness:
                    self.best_solution = self.population[i]
                    self.best_fitness = fitness
            if len(improvement_rates) > 10:
                current_improvement_rate = np.mean(improvement_rates[-10:])
                if current_improvement_rate > self.prev_improvement_rate:
                    self.adaptive_mutation_factor *= 1.1
                else:
                    self.adaptive_mutation_factor *= 0.9
                self.prev_improvement_rate = current_improvement_rate
        return self.best_solution