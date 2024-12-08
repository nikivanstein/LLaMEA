import numpy as np

class ImprovedBatAlgorithmOptimizer:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, func):
        def init_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        def update_frequency(f):
            return f * self.alpha

        def update_loudness(fitness_improved):
            if fitness_improved:
                return self.loudness * self.gamma
            else:
                return self.loudness / self.gamma

        def levy_flight():
            beta = 1.5
            sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                    np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.randn(self.dim) * sigma
            v = np.random.randn(self.dim)
            step = u / abs(v) ** (1 / beta)
            return step

        population = init_population()
        fitness = np.array([func(x) for x in population])
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        for _ in range(self.budget):
            new_population = np.copy(population)

            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    frequency = 0.0
                else:
                    frequency = update_frequency(0.0)
                    new_population[i] += levy_flight() * frequency

                if np.random.rand() < self.loudness and func(new_population[i]) < func(population[i]):
                    population[i] = new_population[i]
                    fitness[i] = func(population[i])
                    if fitness[i] < best_fitness:
                        best_solution = population[i]
                        best_fitness = fitness[i]
                        self.loudness = update_loudness(True)
                    else:
                        self.loudness = update_loudness(False)

        return best_solution