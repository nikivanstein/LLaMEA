class DynamicBatAlgorithmImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.frequency_min = 0
        self.frequency_max = 2
        self.loudness = 0.5
        self.pulse_rate = 0.5
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = float('inf')

    def __call__(self, func):
        chaos_map = LogisticMap(0.4, 3.9, self.budget)  # Example of a chaotic map
        for _ in range(self.budget):
            frequencies = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand(self.population_size)
            mutation_factor = chaos_map.get_next()  # Dynamically adjust mutation factor using a chaotic map
            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    self.velocities[i] += (self.population[i] - self.best_solution) * frequencies[i] * mutation_factor
                self.population[i] += self.velocities[i]
                fitness = func(self.population[i])
                if fitness < self.best_fitness:
                    self.best_solution = self.population[i]
                    self.best_fitness = fitness
        return self.best_solution