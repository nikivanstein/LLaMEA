import numpy as np

class HEAC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.elitism_rate = 0.3
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.population = self.initialize_population()

    def initialize_population(self):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        for i in range(self.population_size):
            population[i] = np.clip(population[i], -5.0, 5.0)
        return population

    def evaluate(self, func):
        fitness = np.array([func(x) for x in self.population])
        self.population = np.array([x for _, x in sorted(zip(fitness, self.population))])
        return fitness

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        for i in range(self.dim):
            if np.random.rand() < self.crossover_rate:
                child[i] = (parent1[i] + parent2[i]) / 2
        return child

    def mutate(self, child):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                child[i] = np.clip(child[i] + np.random.uniform(-1.0, 1.0), -5.0, 5.0)
        return child

    def update(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
            if self.population_size < self.population:
                new_individual = self.crossover(self.population[-1], self.population[np.random.randint(0, self.population_size - 1)])
                new_individual = self.mutate(new_individual)
                self.population = np.array([self.population[-1], new_individual])
            else:
                new_individual = self.crossover(self.population[np.random.randint(0, self.population_size - 1)], self.population[np.random.randint(0, self.population_size - 1)])
                new_individual = self.mutate(new_individual)
                self.population = np.array([self.population[-1], new_individual])
            self.population = np.array([x for _, x in sorted(zip(self.evaluate(func), self.population))])
        return self.population

# Example usage:
def func(x):
    return np.sum(x**2)

heac = HEAC(budget=100, dim=10)
selected_solution = heac.update(func)
print(selected_solution)