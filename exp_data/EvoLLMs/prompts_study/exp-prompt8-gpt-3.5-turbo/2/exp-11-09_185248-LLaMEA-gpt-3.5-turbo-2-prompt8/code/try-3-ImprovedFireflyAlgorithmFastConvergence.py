import numpy as np

class ImprovedFireflyAlgorithmFastConvergence(ImprovedFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.eta = 0.1
        
    def levy_flight(self, evaluations):
        beta = self.beta0 * np.exp(-self.eta * evaluations / self.budget)
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
                        population[i] += self.alpha * np.exp(-self.gamma * np.linalg.norm(population[j] - population[i])) * self.levy_flight(evaluations)
                        evaluations += 1
                        if evaluations >= self.budget:
                            break

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution