import numpy as np

class EnhancedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.2
        self.beta0 = 1.0
        self.gamma = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def adaptive_chaotic_map(self, x, t):
        return 3.9 * x * (1 - x) * np.sin(t)

    def levy_flight(self, beta):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def __call__(self, func):
        population = self.initialize_population()
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if func(population[j]) < func(population[i]):
                        beta = self.beta0 + 0.5 * np.sin(2 * np.pi * evaluations / self.budget)  # Dynamic beta adaptation
                        population[i] += self.alpha * np.exp(-self.gamma * np.linalg.norm(population[j] - population[i])) * self.levy_flight(beta) * self.adaptive_chaotic_map(np.random.random(), evaluations)  # Adaptive chaotic map perturbation
                        evaluations += 1
                        if evaluations >= self.budget:
                            break

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution