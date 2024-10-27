import numpy as np

class HybridFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.alpha = 0.5
        self.beta0 = 1.0
        self.gamma = 0.1
        self.initial_population = np.random.uniform(self.lower_bound, self.upper_bound, (budget, dim))
    
    def local_search(self, solution, func):
        best_solution = solution
        for _ in range(10):
            delta = np.random.uniform(-0.1, 0.1, self.dim)
            new_solution = solution + delta
            new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
            if func(new_solution) < func(best_solution):
                best_solution = new_solution
        return best_solution

    def __call__(self, func):
        population = self.initial_population
        for _ in range(self.budget):
            for i in range(len(population)):
                for j in range(len(population)):
                    if func(population[j]) < func(population[i]):
                        attractiveness = self.beta0 * np.exp(-self.gamma * np.linalg.norm(population[j] - population[i])**2)
                        step = attractiveness * (population[j] - population[i])
                        population[i] += self.alpha * step
                        population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
                population[i] = self.local_search(population[i], func)
        return min(population, key=lambda x: func(x))