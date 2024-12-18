class AcceleratedDynamicBatAlgorithmImproved(AcceleratedDynamicBatAlgorithm):
    def __call__(self, func):
        improvement_rates = []
        elite_size = int(self.population_size * 0.1)  # Top 10% solutions are elite
        for _ in range(self.budget):
            self.pulse_rate = 0.9 / (1 + np.exp(-1 * ((self.budget - _) / self.budget * 10 - 5)))
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
            self.population = sorted(self.population, key=lambda x: func(x))[:elite_size] + self.population[elite_size:]  # Update elite solutions
            if len(improvement_rates) > 10:
                current_improvement_rate = np.mean(improvement_rates[-10:])
                if current_improvement_rate > 0.7:
                    self.adaptive_mutation_factor *= 1.1
                elif current_improvement_rate < 0.3:
                    self.adaptive_mutation_factor *= 0.9
        return self.best_solution
