import numpy as np

class CuckooSearchAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1
        self.nests = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros(self.population_size)

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def evaluate(self, func):
        self.fitnesses = func(self.nests)
        self.fitnesses = np.array(self.fitnesses)
        parents = np.array([self.nests[np.argsort(self.fitnesses)[:int(self.population_size/2)]]])
        offspring = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            if random.random() < self.crossover_probability:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                offspring[i] = (parent1 + parent2) / 2
        offspring = np.array(offspring)
        for i in range(self.population_size):
            if random.random() < self.mutation_probability:
                offspring[i] += np.random.uniform(-1.0, 1.0, self.dim)
        self.nests = offspring

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return np.min(self.nests, axis=0)

# Example usage
def func(x):
    return np.sum(x**2)

csa = CuckooSearchAlgorithm(budget=100, dim=10)
optimal_solution = csa(func)
print(optimal_solution)