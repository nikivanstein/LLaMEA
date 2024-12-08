import numpy as np

class DiversityExplorationTradeoff:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.mutation_rate = 0.1
        self.search_space = np.array([-5.0, 5.0])

    def __call__(self, func):
        for _ in range(self.budget):
            if not self.population:
                self.population.append(np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim))
            else:
                new_individual = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)
                while np.any(np.abs(new_individual - self.population) < 0.1):
                    new_individual = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)
                self.population.append(new_individual)
            func(self.population[-1])
            if len(self.population) > self.budget:
                self.population.pop(0)

    def refine_strategy(self):
        # Refine the strategy by introducing an adaptive mutation rate
        for i in range(len(self.population)):
            if np.random.rand() < self.mutation_rate:
                mutation = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)
                self.population[i] += mutation
                self.population[i] = np.clip(self.population[i], self.search_space[0], self.search_space[1])

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 5
optimizer = DiversityExplorationTradeoff(budget, dim)
optimizer()