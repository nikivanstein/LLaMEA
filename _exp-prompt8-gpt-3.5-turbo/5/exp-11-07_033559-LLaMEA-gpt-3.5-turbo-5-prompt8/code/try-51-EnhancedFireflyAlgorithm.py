import numpy as np
from joblib import Parallel, delayed

class EnhancedFireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def attractiveness_calculation(self, population, i, j):
        r = np.linalg.norm(population[i] - population[j])
        return self.beta * np.exp(-self.gamma * r**2)
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget):
            attractiveness_matrix = np.zeros((self.budget, self.budget))
            for i in range(self.budget):
                for j in range(self.budget):
                    attractiveness_matrix[i, j] = self.attractiveness_calculation(population, i, j)
            
            for i in range(self.budget):
                population += self.alpha * (attractiveness_matrix[:, i].reshape(-1, 1) * (population - population[i])) + np.random.uniform(-1, 1, (self.budget, self.dim))
            
            fitness = Parallel(n_jobs=-1)(delayed(func)(ind) for ind in population)
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        return best_solution