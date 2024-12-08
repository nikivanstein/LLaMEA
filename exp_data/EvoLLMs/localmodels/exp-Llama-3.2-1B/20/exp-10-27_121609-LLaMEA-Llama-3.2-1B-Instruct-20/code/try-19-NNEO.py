import numpy as np

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def evaluate_fitness(individual):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual, individual] + 1e-6:
                self.fitnesses[individual, individual] = fitness
                return individual
            else:
                return None

        for _ in range(self.budget):
            new_individuals = []
            for _ in range(self.population_size):
                new_individual = evaluate_fitness(np.random.uniform(-5.0, 5.0, self.dim))
                if new_individual is not None:
                    new_individuals.append(new_individual)

            new_population = np.random.choice(new_individuals, self.population_size, replace=True)
            new_population = new_population.reshape(-1, self.dim)

            if np.random.rand() < 0.2:
                new_population = self.mutate(new_population)
            if np.random.rand() < 0.2:
                new_population = self.crossover(new_population)

            new_population = new_population.astype(int)
            new_individuals = new_population.flatten().tolist()

            self.population = new_population
            self.fitnesses = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                self.fitnesses[i, new_individuals[i]] = evaluate_fitness(new_individuals[i])

        return self.fitnesses

    def mutate(self, individual):
        mutated_individual = individual.copy()
        if np.random.rand() < self.mutation_rate:
            mutated_individual[np.random.randint(0, self.dim)] = np.random.uniform(-5.0, 5.0)
        return mutated_individual

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            child = np.concatenate((parent1[:np.random.randint(len(parent1)), :], parent2[np.random.randint(len(parent2))]))
            return child
        else:
            child = np.concatenate((parent2[:np.random.randint(len(parent2))], parent1[np.random.randint(len(parent1))]))
            return child