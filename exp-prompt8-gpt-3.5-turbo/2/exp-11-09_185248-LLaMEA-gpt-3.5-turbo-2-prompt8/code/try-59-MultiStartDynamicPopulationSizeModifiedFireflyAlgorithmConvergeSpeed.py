# import numpy as np

class MultiStartDynamicPopulationSizeModifiedFireflyAlgorithmConvergeSpeed:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.beta0 = 1.0
        self.gamma = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def initialize_population(self, population_size):
        return np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))

    def levy_flight(self):
        sigma = (np.math.gamma(1 + self.beta0) * np.sin(np.pi * self.beta0 / 2) / (np.math.gamma((1 + self.beta0) / 2) * self.beta0 * 2 ** ((self.beta0 - 1) / 2))) ** (1 / self.beta0)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v) ** (1 / self.beta0)
        return step

    def __call__(self, func):
        evaluations = 0
        population_size = 50

        while evaluations < self.budget:
            population = self.initialize_population(population_size)
            for i in range(population_size):
                for j in range(population_size):
                    alpha = 0.9 - evaluations / self.budget
                    if func(population[j]) < func(population[i]):
                        population[i] += alpha * np.exp(-self.gamma * np.linalg.norm(population[j] - population[i])) * self.levy_flight()
                        evaluations += 1
                        if evaluations >= self.budget:
                            break
            
            population_size = int(50 * (1 - evaluations / self.budget))  # Dynamic adaptation of population size based on remaining evaluations

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution
