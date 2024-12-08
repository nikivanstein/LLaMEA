import numpy as np
from scipy.optimize import minimize

class MultiFidelityHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.fidelity_levels = 5
        self.population_size = 50
        self.crossover_probability = 0.9
        self.mutation_probability = 0.1
        self.harmony_memory = []
        self.stochastic_search_rate = 0.2

    def __call__(self, func):
        if len(self.harmony_memory) >= self.budget:
            return np.mean([func(x) for x in self.harmony_memory])

        for _ in range(self.population_size):
            if len(self.harmony_memory) < self.budget:
                x = np.random.uniform(-5.0, 5.0, self.dim)
                self.harmony_memory.append(x)
                break

        for _ in range(int(self.population_size * (1 - self.stochastic_search_rate))):
            if len(self.harmony_memory) < self.budget:
                parent1, parent2 = np.random.choice(self.harmony_memory, 2, replace=False)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                self.harmony_memory.append(child)

        for _ in range(int(self.population_size * self.stochastic_search_rate)):
            x = np.random.uniform(-5.0, 5.0, self.dim)
            self.harmony_memory.append(x)

        return np.mean([func(x) for x in self.harmony_memory])

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_probability:
            child = parent1 + parent2 - parent1 * parent2
            return child
        else:
            return parent1

    def mutate(self, child):
        if np.random.rand() < self.mutation_probability:
            index = np.random.randint(0, self.dim)
            child[index] += np.random.uniform(-1.0, 1.0)
            if child[index] < -5.0:
                child[index] = -5.0
            elif child[index] > 5.0:
                child[index] = 5.0

# Example usage:
def func(x):
    return np.sin(x[0]) * np.cos(x[1]) * np.exp(x[2])

ms = MultiFidelityHarmonySearch(budget=100, dim=3)
result = ms(func)
print(result)