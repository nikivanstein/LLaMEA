import numpy as np

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            new_individual = individual.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.2:
                    new_individual[i] += np.random.uniform(-5.0, 5.0)
            return new_individual

        def crossover(parent1, parent2):
            if np.random.rand() < 0.5:
                child = parent1.copy()
                for i in range(self.dim):
                    child[i] = parent2[i]
                return child
            else:
                child = parent2.copy()
                for i in range(self.dim):
                    child[i] = parent1[i]
                return child

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        # Select the fittest individuals
        fittest_individuals = np.argsort(self.fitnesses, axis=1)[:, :self.population_size // 2]

        # Select parents using roulette wheel selection
        parents = np.random.choice(fittest_individuals, size=self.population_size, replace=True, p=self.fitnesses / self.fitnesses.sum())

        # Select children using crossover and mutation
        children = np.array([crossover(parents[i], parents[(i+1)%self.population_size]) for i in range(self.population_size)])

        # Mutate the children
        children = np.array([mutate(child) for child in children])

        # Replace the old population with the new one
        self.population = children

# One-line description: Novel NNEO Algorithm using roulette wheel selection and crossover/mutation strategies.
# Code: 