import numpy as np

class DynamicAdaptivePopulationBatDEAlgorithm(DynamicAdaptivePopulationBatAlgorithm):
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.1, freq_min=0.1, freq_max=1.0, loudness_range=(0.2, 0.8), pulse_rate_range=(0.1, 0.9), de_cr=0.9, de_f=0.8):
        super().__init__(budget, dim, population_size, loudness, pulse_rate, alpha, gamma, freq_min, freq_max, loudness_range, pulse_rate_range)
        self.de_cr = de_cr
        self.de_f = de_f

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
                    idxs = np.random.choice([idx for idx in range(self.population_size) if idx != i], 2, replace=False)
                    x_r1, x_r2 = population[idxs]
                    mutant_vector = population[i] + self.de_f * (x_r1 - x_r2)
                    crossover_mask = np.random.rand(self.dim) < self.de_cr
                    new_solution = np.where(crossover_mask, mutant_vector, population[i])

                new_fitness = func(new_solution)
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
                    if new_fitness < func(best_individual):
                        best_individual = new_solution

                t += 1
                if t >= self.budget:
                    break

            # Adaptive population size
            if np.random.rand() < 0.1:  # 10% chance to adjust population size
                self.population_size = max(2, min(100, int(self.population_size * np.random.uniform(0.9, 1.1))))

            # Dynamic adaptation of algorithmic parameters
            self.alpha *= 0.999
            self.gamma *= 1.001
            self.freq_min *= 1.001
            self.freq_max *= 0.999
            self.loudness = max(0.1, min(0.9, self.loudness * np.random.uniform(0.9, 1.1)))
            self.pulse_rate = max(0.1, min(0.9, self.pulse_rate * np.random.uniform(0.9, 1.1)))

        return best_individual