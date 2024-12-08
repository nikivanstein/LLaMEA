import numpy as np

class FireflyAlgorithmOptimization:
    def __init__(self, budget, dim, alpha=0.5, beta0=1.0, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def __call__(self, func):
        def attractiveness(distance):
            return self.beta0 * np.exp(-self.gamma * distance**2)

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

        for _ in range(self.budget):
            new_population = np.zeros_like(population)

            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if func(population[j]) < func(population[i]):
                        distance = np.linalg.norm(population[j] - population[i])
                        attractiveness_factor = attractiveness(distance)
                        new_population[i] += attractiveness_factor * (population[j] - population[i])

            for i in range(self.pop_size):
                new_population[i] += self.alpha * np.random.uniform(-1, 1, self.dim)
                new_population[i] = np.clip(new_population[i], -5.0, 5.0)

            population = new_population

        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution