class ImprovedAcceleratedDynamicBatAlgorithmConverge(AcceleratedDynamicBatAlgorithmImproved):
    def __call__(self, func):
        improvement_rates = []
        dynamic_factor_adjustment = 0.1
        for _ in range(self.budget):
            self.pulse_rate = 0.9 / (1 + np.exp(-1 * ((_ - self.budget) / self.budget * 10 - 5)))
            frequencies = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand(self.population_size)
            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    self.velocities[i] += (self.population[i] - self.best_solution) * frequencies[i] * self.adaptive_mutation_factor
                self.population[i] += self.velocities[i] * dynamic_factor_adjustment
                fitness = func(self.population[i])
                improvement_rate = (self.best_fitness - fitness) / self.best_fitness
                improvement_rates.append(improvement_rate)
                if fitness < self.best_fitness:
                    self.best_solution = self.population[i]
                    self.best_fitness = fitness
            if len(improvement_rates) > 10:
                current_improvement_rate = np.mean(improvement_rates[-10:])
                dynamic_factor_adjustment = 0.1 + (current_improvement_rate - 0.5) * 0.1
                if current_improvement_rate > 0.6:
                    self.adaptive_mutation_factor *= 1.2
                    if self.population_size < 50:  # Adaptive population size adjustment
                        self.population_size += 5
                        self.population = np.append(self.population, np.random.uniform(-5, 5, (5, self.dim)), axis=0)
                        self.velocities = np.append(self.velocities, np.zeros((5, self.dim)), axis=0)
                elif current_improvement_rate < 0.4:
                    self.adaptive_mutation_factor *= 0.8
                    if self.population_size > 10:  # Adaptive population size adjustment
                        self.population_size -= 5
                        self.population = self.population[:self.population_size]
                        self.velocities = self.velocities[:self.population_size]
                if improvement_rate > 0.8:  # Adaptive step size adjustment
                    dynamic_factor_adjustment = min(dynamic_factor_adjustment * 1.2, 0.8)
                if current_improvement_rate > 0.7:  # Faster convergence adjustment
                    dynamic_factor_adjustment = min(dynamic_factor_adjustment * 1.5, 0.8)
        return self.best_solution