import numpy as np

class CrossoverPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.warmup = 10
        self.crossover_prob = 0.7
        self.mutation_prob = 0.1
        self.population = self.initialize_population()

    def initialize_population(self):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([self.func(x) for x in population])
        return population, fitness

    def func(self, x):
        # Black box function to optimize
        return np.sum(x**2)

    def evaluate(self, population):
        fitness = np.array([self.func(x) for x in population])
        return fitness

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_prob:
            child = parent1 + parent2 - parent1 * parent2
            return child
        else:
            return parent1

    def mutate(self, x):
        if np.random.rand() < self.mutation_prob:
            idx = np.random.randint(0, self.dim)
            x[idx] += np.random.uniform(-1.0, 1.0)
        return x

    def update(self):
        # Update the population using PSO
        for i in range(self.population_size):
            r1 = np.random.uniform(0.0, 1.0)
            r2 = np.random.uniform(0.0, 1.0)
            v1 = self.population[i] + r1 * (self.population[i] - self.population[np.argmax(self.evaluate(self.population))])
            v2 = self.population[i] + r2 * (self.population[np.argmin(self.evaluate(self.population))] - self.population[i])
            self.population[i] = self.crossover(v1, v2)
            self.population[i] = self.mutate(self.population[i])

    def optimize(self):
        for _ in range(self.budget):
            self.update()
            fitness = self.evaluate(self.population)
            if np.min(fitness) < np.min(self.evaluate(self.population)):
                self.population, self.population = self.population, self.population[np.argmin(fitness)]
            if _ > self.warmup:
                self.population = self.population[np.random.choice(self.population_size)]
        return self.population[np.argmin(self.evaluate(self.population))]

# Example usage
budget = 100
dim = 10
optimizer = CrossoverPSO(budget, dim)
best_solution = optimizer.optimize()
print(best_solution)