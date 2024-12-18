class DynamicBatAlgorithmImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.frequency_min = 0
        self.frequency_max = 2
        self.loudness = 0.5
        self.pulse_rate = 0.5
        self.mutation_factor = 0.1
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = float('inf')

    def __call__(self, func):
        last_fitness = float('inf')
        for _ in range(self.budget):
            frequencies = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand(self.population_size)
            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    fitness_before = func(self.population[i])
                    self.velocities[i] += (self.population[i] - self.best_solution) * frequencies[i] * self.mutation_factor
                    self.population[i] += self.velocities[i]
                    fitness_after = func(self.population[i])
                    fitness_improvement_rate = (last_fitness - fitness_after) / last_fitness
                    self.mutation_factor *= 1.1 if fitness_improvement_rate > 0 else 0.9
                    last_fitness = fitness_after
                    if fitness_after < self.best_fitness:
                        self.best_solution = self.population[i]
                        self.best_fitness = fitness_after
        return self.best_solution