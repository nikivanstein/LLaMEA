import numpy as np

class HybridBDHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iterations = budget // self.population_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.f_min = 0.0
        self.f_max = 2.0
        self.loudness = 0.9
        self.pulse_rate = 0.5
        self.hmcr = 0.7
        self.par = 0.5
        self.mutation_rate = 0.2

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

        def levy_flight(beta=1.5):
            sigma_u = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            sigma_v = 1
            u = np.random.normal(0, sigma_u, self.dim)
            v = np.random.normal(0, sigma_v, self.dim)
            step = u / np.abs(v) ** (1 / beta)
            return step

        population = initialize_population()
        fitness_values = [func(individual) for individual in population]
        best_individual = population[np.argmin(fitness_values)]
        best_fitness = np.min(fitness_values)

        for _ in range(self.max_iterations):
            new_population = []
            for idx, individual in enumerate(population):
                if np.random.rand() < self.loudness:
                    y = individual + levy_flight() * (best_individual - individual)
                    for j in range(self.dim):
                        if np.random.rand() < self.pulse_rate:
                            y[j] = best_individual[j] + np.random.uniform(-1, 1) * (best_individual[j] - individual[j])
                    y = np.clip(y, self.lower_bound, self.upper_bound)
                    if func(y) < fitness_values[idx]:
                        new_population.append(y)
                        fitness_values[idx] = func(y)
                        if func(y) < best_fitness:
                            best_individual = y
                            best_fitness = func(y)
                else:
                    new_population.append(individual)
            population = np.array(new_population)

        return best_individual