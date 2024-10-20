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

    def __call__(self, func):
        if len(self.harmony_memory) >= self.budget:
            return np.mean([func(x) for x in self.harmony_memory])

        for _ in range(self.population_size):
            if len(self.harmony_memory) < self.budget:
                x = np.random.uniform(-5.0, 5.0, self.dim)
                self.harmony_memory.append(x)
                break

        for _ in range(self.population_size):
            if len(self.harmony_memory) < self.budget:
                parent1, parent2 = np.random.choice(self.harmony_memory, 2, replace=False)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                self.harmony_memory.append(child)

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

class DifferentialEvolutionWithFidelity:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.fidelity_levels = 5
        self.population_size = 50
        self.crossover_probability = 0.9
        self.mutation_probability = 0.1
        self.fidelity = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        if len(self.fidelity) >= self.budget:
            return np.mean([func(x) for x in self.fidelity])

        for _ in range(self.population_size):
            if len(self.fidelity) < self.budget:
                x = np.random.uniform(-5.0, 5.0, self.dim)
                self.fidelity[len(self.fidelity)-1] = x
                break

        for _ in range(self.population_size):
            if len(self.fidelity) < self.budget:
                parent1, parent2 = np.random.choice(self.fidelity, 2, replace=False)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                self.fidelity[len(self.fidelity)-1] = child

        return np.mean([func(x) for x in self.fidelity])

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_probability:
            child = parent1 + parent2 - parent1 * parent2
            return child
        else:
            return parent1

    def mutate(self, child):
        if np.random.rand() < self.mutation_probability:
            index1 = np.random.randint(0, self.dim)
            index2 = np.random.randint(0, self.dim)
            index3 = np.random.randint(0, self.dim)
            child[index1] += self.fidelity[index2] - self.fidelity[index3]
            if child[index1] < -5.0:
                child[index1] = -5.0
            elif child[index1] > 5.0:
                child[index1] = 5.0

# Example usage:
def func(x):
    return np.sin(x[0]) * np.cos(x[1]) * np.exp(x[2])

ms = MultiFidelityHarmonySearch(budget=100, dim=3)
de = DifferentialEvolutionWithFidelity(budget=100, dim=3)
result_ms = ms(func)
result_de = de(func)
print(result_ms)
print(result_de)