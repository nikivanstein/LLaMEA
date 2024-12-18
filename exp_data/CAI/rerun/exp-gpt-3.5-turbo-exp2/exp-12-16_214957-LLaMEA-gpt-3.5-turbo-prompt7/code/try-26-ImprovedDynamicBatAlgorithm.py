class ImprovedDynamicBatAlgorithm(AcceleratedDynamicBatAlgorithm):
    def __call__(self, func):
        inertia_weights = []
        for _ in range(self.budget):
            self.pulse_rate = 0.9 / (1 + np.exp(-1 * ((self.budget - _) / self.budget * 10 - 5)))
            frequencies = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand(self.population_size)
            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    inertia_weight = 0.5 + 0.4 * (self.budget - _) / self.budget
                    self.velocities[i] = inertia_weight * self.velocities[i] + (self.population[i] - self.best_solution) * frequencies[i] * self.adaptive_mutation_factor
                self.population[i] += self.velocities[i]
                fitness = func(self.population[i])
                improvement_rate = (self.best_fitness - fitness) / self.best_fitness
                inertia_weights.append(inertia_weight)
                if fitness < self.best_fitness:
                    self.best_solution = self.population[i]
                    self.best_fitness = fitness
            if len(inertia_weights) > 10:
                current_inertia_weight = np.mean(inertia_weights[-10:])
                if current_inertia_weight > 0.7:
                    self.adaptive_mutation_factor *= 1.1
                elif current_inertia_weight < 0.3:
                    self.adaptive_mutation_factor *= 0.9
        return self.best_solution