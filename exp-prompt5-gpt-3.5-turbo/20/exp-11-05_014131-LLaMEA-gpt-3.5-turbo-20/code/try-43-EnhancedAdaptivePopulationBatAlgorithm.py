import numpy as np

class EnhancedAdaptivePopulationBatAlgorithm(AdaptivePopulationBatAlgorithm):
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_individual = population[np.argmin(fitness)]

        t = 0
        while t < self.budget:
            for i in range(self.population_size):
                frequency = self.freq_min + (self.freq_max - self.freq_min) * np.random.rand()
                velocity = population[i] + (population[i] - best_individual) * self.alpha + frequency * self.gamma
                new_solution = population[i] + velocity
                if np.random.rand() < self.pulse_rate:
                    new_loudness = np.random.uniform(self.loudness_range[0], self.loudness_range[1])
                    new_pulse_rate = np.random.uniform(self.pulse_rate_range[0], self.pulse_rate_range[1])
                    new_solution = best_individual + np.random.uniform(-1, 1, self.dim) * new_loudness
                else:
                    levy = 0.001 * np.random.standard_cauchy(self.dim)
                    new_solution = population[i] + levy

                new_fitness = func(new_solution)
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
                    if new_fitness < func(best_individual):
                        best_individual = new_solution

                t += 1
                if t >= self.budget:
                    break

            # Dynamic adaptive parameter adjustments
            self.freq_max = max(self.freq_min, min(1.0, self.freq_max + np.random.uniform(-0.1, 0.1)))
            self.alpha = max(0.1, min(0.9, self.alpha + np.random.uniform(-0.05, 0.05)))
            self.pulse_rate = max(0.1, min(0.9, self.pulse_rate + np.random.uniform(-0.05, 0.05)))
            
            if np.random.rand() < 0.1:  # 10% chance to adjust population size
                self.population_size = max(2, min(100, int(self.population_size * np.random.uniform(0.9, 1.1))))

        return best_individual