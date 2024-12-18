from joblib import Parallel, delayed

class ImprovedParallelAcceleratedDynamicBatAlgorithm(AcceleratedDynamicBatAlgorithmImproved):
    def __call__(self, func):
        improvement_rates = []
        step_size = 0.1
        prev_best_fitness = np.inf
        for _ in range(self.budget):
            self.pulse_rate = 0.9 / (1 + np.exp(-1 * ((_ - self.budget) / self.budget * 10 - 5)))
            frequencies = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand(self.population_size)
            def evaluate_fitness(idx):
                if np.random.rand() > self.pulse_rate:
                    self.velocities[idx] += (self.population[idx] - self.best_solution) * frequencies[idx] * self.adaptive_mutation_factor
                self.population[idx] += self.velocities[idx] * step_size
                fitness = func(self.population[idx])
                return idx, fitness
            results = Parallel(n_jobs=-1)(delayed(evaluate_fitness)(i) for i in range(self.population_size))
            for idx, fitness in results:
                improvement_rate = (self.best_fitness - fitness) / self.best_fitness
                improvement_rates.append(improvement_rate)
                if fitness < self.best_fitness:
                    self.best_solution = self.population[idx]
                    self.best_fitness = fitness
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
                if improvement_rate > 0.8:
                    step_size = min(step_size * 1.2, 0.8)
                prev_best_fitness = self.best_fitness
        return self.best_solution