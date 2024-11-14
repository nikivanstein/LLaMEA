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
            for i in range(self.budget):
                j = np.random.randint(self.budget)
                r = np.linalg.norm(population[i] - population[j])
                attractiveness = self.beta * np.exp(-self.gamma * r**2)
                new_solution = population[i] + self.alpha * (attractiveness * (population[j] - population[i])) + np.random.uniform(-1, 1, self.dim)
                new_fitness = func(new_solution)
                
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    population[i] = new_solution
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        return best_solution