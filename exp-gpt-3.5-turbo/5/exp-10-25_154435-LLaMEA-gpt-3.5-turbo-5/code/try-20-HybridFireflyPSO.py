import numpy as np

class HybridFireflyPSO:
    def __init__(self, budget, dim, population_size=50, alpha=0.1, beta0=1.0, gamma=1.0, w=0.5, c1=2.0, c2=2.0):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        best_solution = population[np.argmin([func(individual) for individual in population])
        for _ in range(self.budget):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if func(population[i]) < func(population[j]):
                        attractiveness = self.beta0 * np.exp(-self.gamma * np.linalg.norm(population[i] - population[j])**2)
                        step = self.alpha * (np.random.rand(self.dim) - 0.5)
                        new_position = population[i] + attractiveness * (population[j] - population[i]) + step
                        population[i] = new_position
                velocity = self.w * velocity + self.c1 * np.random.rand(self.dim) * (best_solution - population[i]) + self.c2 * np.random.rand(self.dim) * (population[i] - best_solution)
                population[i] = population[i] + velocity

        return best_solution