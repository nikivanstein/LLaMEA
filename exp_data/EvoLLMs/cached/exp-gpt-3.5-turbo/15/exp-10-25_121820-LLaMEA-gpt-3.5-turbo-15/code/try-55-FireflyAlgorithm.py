import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim, population_size=30, alpha=0.5, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def attractiveness(self, brightness, distance):
        return self.beta0 * np.exp(-self.gamma * distance**2) + self.alpha * brightness

    def move_firefly(self, firefly, brightest_firefly):
        distance = np.linalg.norm(firefly - brightest_firefly)
        attractiveness = self.attractiveness(firefly_fitness, distance)
        return firefly + attractiveness * (brightest_firefly - firefly) + 0.01 * np.random.normal(size=self.dim)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        firefly_fitness = np.array([func(individual) for individual in population])
        brightest_idx = np.argmin(firefly_fitness)
        brightest_firefly = population[brightest_idx]

        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if firefly_fitness[j] < firefly_fitness[i]:
                        population[i] = self.move_firefly(population[i], population[j])

            firefly_fitness = np.array([func(individual) for individual in population])
            new_brightest_idx = np.argmin(firefly_fitness)
            if firefly_fitness[new_brightest_idx] < firefly_fitness[brightest_idx]:
                brightest_firefly = population[new_brightest_idx]
                brightest_idx = new_brightest_idx

        return brightest_firefly