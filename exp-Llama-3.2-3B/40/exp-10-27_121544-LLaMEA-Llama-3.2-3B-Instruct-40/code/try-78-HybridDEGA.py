import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import randint

class HybridDEGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        for _ in range(int(self.budget * 0.6)):
            population_size = 50
            population = np.random.uniform(self.bounds[0][0], self.bounds[0][1], (population_size, self.dim))
            for _ in range(10):
                new_population = []
                for _ in range(population_size):
                    parent1 = np.random.choice(population, 1, p=None)[0]
                    parent2 = np.random.choice(population, 1, p=None)[0]
                    child = (parent1 + parent2) / 2
                    new_population.append(child)
                population = new_population

        best_individual = np.min(population, axis=0)
        return func(best_individual), best_individual

# Example usage:
if __name__ == "__main__":
    func = lambda x: x[0]**2 + 2*x[1]**2
    hybridDEGA = HybridDEGA(10, 2)
    f_min, x_min = hybridDEGA(func)
    print(f"f_min: {f_min}, x_min: {x_min}")