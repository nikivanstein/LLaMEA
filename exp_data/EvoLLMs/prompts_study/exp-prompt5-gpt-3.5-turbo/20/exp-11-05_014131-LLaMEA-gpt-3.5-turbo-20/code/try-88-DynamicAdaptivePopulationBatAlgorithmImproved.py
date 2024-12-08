import numpy as np

class DynamicAdaptivePopulationBatAlgorithmImproved:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.1, freq_min=0.1, freq_max=1.0, loudness_range=(0.2, 0.8), pulse_rate_range=(0.1, 0.9), mutation_rate=0.1, crossover_rate=0.2, mutation_range=(0.1, 0.5)):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.alpha = alpha
        self.gamma = gamma
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.loudness_range = loudness_range
        self.pulse_rate_range = pulse_rate_range
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_range = mutation_range

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
                    if np.random.rand() < self.crossover_rate:
                        parent_indices = np.random.choice(self.population_size, 2, replace=False)
                        parent1, parent2 = population[parent_indices]
                        crossover_point = np.random.randint(1, self.dim)
                        new_solution = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                    else:
                        if np.random.rand() < self.mutation_rate:
                            mutation = np.random.uniform(-self.mutation_range, self.mutation_range, self.dim)
                            new_solution = population[i] + mutation
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

            if np.random.rand() < 0.1:
                self.population_size = max(2, min(100, int(self.population_size * np.random.uniform(0.9, 1.1))))

            self.alpha *= 0.999
            self.gamma *= 1.001
            self.freq_min *= 1.001
            self.freq_max *= 0.999
            self.loudness = max(0.1, min(0.9, self.loudness * np.random.uniform(0.9, 1.1)))
            self.pulse_rate = max(0.1, min(0.9, self.pulse_rate * np.random.uniform(0.9, 1.1)))

        return best_individual