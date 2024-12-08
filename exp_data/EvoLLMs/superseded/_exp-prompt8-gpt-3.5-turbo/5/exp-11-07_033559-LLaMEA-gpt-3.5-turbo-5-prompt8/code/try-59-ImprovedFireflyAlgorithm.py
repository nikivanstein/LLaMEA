import numpy as np

class ImprovedFireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget):
            i = np.arange(self.budget)
            j = np.arange(self.budget)

            less_fit_mask = fitness[j] < fitness[i]
            i_less_fit = i[less_fit_mask]

            r = np.linalg.norm(population[i_less_fit] - population[j[:, None]], axis=2)
            attractiveness = self.beta * np.exp(-self.gamma * r**2)
            population[i_less_fit] += self.alpha * (attractiveness[:,:,None] * (population[j] - population[i_less_fit])) + np.random.uniform(-1, 1, (len(i_less_fit), self.dim))
            fitness[i_less_fit] = [func(population[idx]) for idx in i_less_fit]

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        return best_solution