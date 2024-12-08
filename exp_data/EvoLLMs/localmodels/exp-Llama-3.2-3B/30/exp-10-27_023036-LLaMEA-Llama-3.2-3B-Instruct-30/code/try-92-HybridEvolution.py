import numpy as np

class HybridEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_prob = 0.3
        self.population = self.initialize_population()

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def evaluate_function(self, func):
        fitness = np.array([func(x) for x in self.population.flatten()])
        fitness = np.array([fitness.min(), fitness.max(), np.mean(fitness), np.median(fitness)])
        return fitness

    def select_parents(self, fitness):
        parents = np.random.choice(self.population_size, size=2, p=fitness)
        return self.population[parents].flatten()

    def mutate(self, x):
        if np.random.rand() < self.mutation_prob:
            index = np.random.randint(0, self.dim)
            x[index] += np.random.uniform(-1.0, 1.0)
        return x

    def evolve(self, func):
        fitness = self.evaluate_function(func)
        parents = self.select_parents(fitness)
        children = np.array([self.mutate(parent) for parent in parents])
        self.population = np.vstack((self.population, children))
        self.population = self.population[:self.population_size]
        return fitness

    def optimize(self, func):
        for _ in range(self.budget):
            fitness = self.evolve(func)
            best_idx = np.argmin(fitness)
            best_x = self.population[best_idx]
            print(f'Iteration {_+1}, Best x: {best_x}, Best fitness: {fitness[best_idx]}')

# Example usage
def func(x):
    return np.sum(x**2)

hybrid = HybridEvolution(100, 10)
hybrid.optimize(func)