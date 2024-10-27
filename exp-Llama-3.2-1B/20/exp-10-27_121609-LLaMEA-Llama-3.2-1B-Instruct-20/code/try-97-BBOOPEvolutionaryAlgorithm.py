import numpy as np

class BBOOPEvolutionaryAlgorithm:
    def __init__(self, budget, dim, mutation_rate, bounds, mutation_strategy):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.bounds = bounds
        self.mutation_strategy = mutation_strategy
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            if np.random.rand() < self.mutation_rate:
                index, value = np.random.choice(self.dim, 2, replace=False)
                individual[index] = self.mutation_strategy(individual[index], individual, bounds, index, value)
            return individual

        def select_parents(population):
            fitnesses = self.fitnesses
            parents = np.random.choice(len(population), self.population_size, replace=False, p=fitnesses / self.fitnesses)
            return parents, fitnesses

        for _ in range(self.budget):
            new_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                individual = self.population[i]
                fitness = objective(individual)
                if fitness < self.fitnesses[i, individual] + 1e-6:
                    new_population[i] = individual
                    self.fitnesses[i, individual] = fitness

            new_population = np.array([mutate(individual) for individual in new_population])
            parents, fitnesses = select_parents(new_population)

            self.population = np.concatenate((self.population, new_population), axis=0)
            self.population = np.array([individual for individual in self.population if individual not in parents])

            self.population = np.array([individual for individual in self.population if individual not in parents])
            self.population = np.concatenate((self.population, new_population), axis=0)

        return self.fitnesses

# One-line description with the main idea
# BBOOPEvolutionaryAlgorithm: An evolutionary algorithm for black box optimization
# Parameters:
#   budget: Maximum number of function evaluations
#   dim: Dimensionality of the search space
#   mutation_rate: Probability of mutation in the population
#   bounds: Search space bounds
#   mutation_strategy: Mutation strategy for the population