from concurrent.futures import ThreadPoolExecutor

class ParallelAcceleratedDynamicBatAlgorithm(AcceleratedDynamicBatAlgorithm):
    def __call__(self, func):
        improvement_rates = []
        with ThreadPoolExecutor() as executor:
            for _ in range(self.budget):
                self.pulse_rate = 0.9 / (1 + np.exp(-1 * ((self.budget - _) / self.budget * 10 - 5)))
                frequencies = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand(self.population_size)
                futures = [executor.submit(self._update_population, i, func, frequencies) for i in range(self.population_size)]
                for future in futures:
                    improvement_rate, fitness = future.result()
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
        return self.best_solution

    def _update_population(self, i, func, frequencies):
        if np.random.rand() > self.pulse_rate:
            self.velocities[i] += (self.population[i] - self.best_solution) * frequencies[i] * self.adaptive_mutation_factor
        self.population[i] += self.velocities[i]
        fitness = func(self.population[i])
        improvement_rate = (self.best_fitness - fitness) / self.best_fitness
        return improvement_rate, fitness