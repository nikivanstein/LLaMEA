from joblib import Parallel, delayed

class ParallelDynamicBatAlgorithm(DynamicBatAlgorithm):
    def __call__(self, func):
        for _ in range(self.budget):
            frequencies = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand(self.population_size)
            results = Parallel(n_jobs=-1)(delayed(self._update_bat)(i, func, frequencies) for i in range(self.population_size))
            for i, fitness in results:
                if fitness < self.best_fitness:
                    self.best_solution = self.population[i]
                    self.best_fitness = fitness
        return self.best_solution
    
    def _update_bat(self, i, func, frequencies):
        if np.random.rand() > self.pulse_rate:
            self.velocities[i] += (self.population[i] - self.best_solution) * frequencies[i] * self.mutation_factor
        self.population[i] += self.velocities[i]
        fitness = func(self.population[i])
        return i, fitness