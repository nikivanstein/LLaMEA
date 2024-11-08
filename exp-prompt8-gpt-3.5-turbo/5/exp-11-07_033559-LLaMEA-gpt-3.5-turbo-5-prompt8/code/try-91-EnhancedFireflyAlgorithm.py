import numpy as np

class EnhancedFireflyAlgorithm:
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
            attractiveness_matrix = self.beta * np.exp(-self.gamma * np.linalg.norm(population[:, None] - population, axis=2)**2)
            np.fill_diagonal(attractiveness_matrix, 0)
            population += self.alpha * (np.sum(attractiveness_matrix[:, :, None] * (population[:, None] - population), axis=1)) + np.random.uniform(-1, 1, (self.budget, self.dim))
            fitness = np.array([func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        return best_solution