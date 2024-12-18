import concurrent.futures

class ParallelAcceleratedDynamicBatAlgorithmImproved(AcceleratedDynamicBatAlgorithmImproved):
    def __call__(self, func):
        improvement_rates = []
        step_size = 0.1
        prev_best_fitness = np.inf
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for _ in range(self.budget):
                self.pulse_rate = 0.9 / (1 + np.exp(-1 * ((_ - self.budget) / self.budget * 10 - 5)))
                frequencies = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand(self.population_size)
                candidate_results = list(executor.map(self.update_candidate, range(self.population_size), [func]*self.population_size, frequencies))
                for i, candidate_result in candidate_results:
                    i, candidate_fitness, improvement_rate = candidate_result
                    improvement_rates.append(improvement_rate)
                    if candidate_fitness < self.best_fitness:
                        self.best_solution = self.population[i]
                        self.best_fitness = candidate_fitness
                if len(improvement_rates) > 10:
                    current_improvement_rate = np.mean(improvement_rates[-10:])
                    if current_improvement_rate > 0.6:
                        self.adaptive_mutation_factor *= 1.2
                        step_size = min(step_size * 1.1, 0.5)
                        if self.population_size < 50:
                            self.population_size += 5
                            self.population = np.append(self.population, np.random.uniform(-5, 5, (5, self.dim)), axis=0)
                            self.velocities = np.append(self.velocities, np.zeros((5, self.dim)), axis=0)
                    elif current_improvement_rate < 0.4:
                        self.adaptive_mutation_factor *= 0.8
                        step_size = max(step_size * 0.9, 0.05)
                        if self.population_size > 10:
                            self.population_size -= 5
                            self.population = self.population[:self.population_size]
                            self.velocities = self.velocities[:self.population_size]

                    if self.best_fitness < prev_best_fitness:
                        self.adaptive_mutation_factor *= 1.1
                    prev_best_fitness = self.best_fitness
        return self.best_solution

    def update_candidate(self, i, func, frequency):
        if np.random.rand() > self.pulse_rate:
            self.velocities[i] += (self.population[i] - self.best_solution) * frequency * self.adaptive_mutation_factor
        self.population[i] += self.velocities[i] * 0.1
        fitness = func(self.population[i])
        improvement_rate = (self.best_fitness - fitness) / self.best_fitness
        return i, fitness, improvement_rate
