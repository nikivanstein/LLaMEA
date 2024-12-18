class AdaptiveDynamicBatAlgorithm(DynamicBatAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.adaptive_mutation_factor = 0.1
        self.dynamic_population_size = self.population_size

    def __call__(self, func):
        improvement_rates = []
        for _ in range(self.budget):
            frequencies = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand(self.dynamic_population_size)
            for i in range(self.dynamic_population_size):
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
                if current_improvement_rate > 0.7:
                    self.adaptive_mutation_factor *= 1.1
                elif current_improvement_rate < 0.3:
                    self.adaptive_mutation_factor *= 0.9
            if np.random.rand() < 0.1:  # Randomly adjust population size
                self.dynamic_population_size = max(1, min(self.dynamic_population_size + np.random.choice([-1, 1]), self.population_size))
        return self.best_solution