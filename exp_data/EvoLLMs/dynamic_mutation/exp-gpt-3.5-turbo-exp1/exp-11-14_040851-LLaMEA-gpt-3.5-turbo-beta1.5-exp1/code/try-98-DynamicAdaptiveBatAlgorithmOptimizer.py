import numpy as np

class DynamicAdaptiveBatAlgorithmOptimizer(EnhancedBatAlgorithmOptimizer):
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.9, differential_weight=0.5, crossover_rate=0.7, crossover_adjust_rate=0.1, mutation_scale=0.1, global_mutation_scale=0.1):
        super().__init__(budget, dim, population_size, loudness, pulse_rate, alpha, gamma, differential_weight, crossover_rate, crossover_adjust_rate, mutation_scale)
        self.global_mutation_scale = global_mutation_scale

    def __call__(self, func):
        def update_global_mutation():
            return np.random.normal(0, self.global_mutation_scale, self.dim)

        population = self.init_population()
        fitness = np.array([func(x) for x in population])
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        for _ in range(self.budget):
            new_population = self.differential_evolution(population, fitness, func)
            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    frequency = 0.0
                else:
                    frequency = self.update_frequency(0.0)
                    new_population[i] += self.levy_flight() * frequency

                if np.random.rand() < self.loudness and func(new_population[i]) < func(population[i]):
                    population[i] = new_population[i]
                    fitness[i] = func(population[i])
                    if fitness[i] < best_fitness:
                        best_solution = population[i]
                        best_fitness = fitness[i]
                        self.loudness = self.update_loudness(True)
                    else:
                        self.loudness = self.update_loudness(False)

                # Introducing dynamic adaptive mutation
                if np.random.rand() < 0.5:  # Global mutation
                    mutation = update_global_mutation()
                else:  # Individual mutation
                    mutation = np.random.normal(0, self.mutation_scale, self.dim)
                new_population[i] += mutation

            if _ % int(0.2 * self.budget) == 0:
                mean_fitness = np.mean(fitness)
                std_fitness = np.std(fitness)
                if std_fitness < 0.1:
                    self.crossover_rate += self.crossover_adjust_rate
                elif std_fitness > 0.5:
                    self.crossover_rate -= self.crossover_adjust_rate
                self.crossover_rate = np.clip(self.crossover_rate, 0, 1)

        return best_solution