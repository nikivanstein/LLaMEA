import numpy as np

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.mutation_rate = 0.1

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.mutation_rate = 0.1
        self.population_size = 100
        self.elite_size = 20

    def __call__(self, func):
        # Initialize population with random solutions
        population = self.initialize_population()

        # Evolve population over iterations
        for _ in range(100):
            # Select elite individuals
            elite = self.select_elite(population)

            # Perform crossover and mutation
            offspring = self.crossover(elite)
            offspring = self.mutate(offspring)

            # Replace worst individuals with offspring
            population = self.replace_worst(population, elite, offspring)

        # Evaluate fitness of best individuals
        fitness = self.evaluate_fitness(population)

        # Return best individual
        return population[np.argmax(fitness)]

    def initialize_population(self):
        # Initialize population with random solutions
        population = np.random.rand(self.population_size, self.dim)
        return population

    def select_elite(self, population):
        # Select elite individuals based on fitness
        elite = population[np.argsort(-self.evaluate_fitness(population))]
        return elite[:self.elite_size]

    def crossover(self, elite):
        # Perform crossover between elite individuals
        offspring = elite.copy()
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                # Perform mutation
                idx = np.random.randint(0, self.dim)
                offspring[i, idx] += np.random.uniform(-1, 1)
        return offspring

    def mutate(self, offspring):
        # Perform mutation on offspring
        mutated = offspring.copy()
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                # Perform mutation
                mutated[i, np.random.randint(0, self.dim)] += np.random.uniform(-1, 1)
        return mutated

    def replace_worst(self, population, elite, offspring):
        # Replace worst individuals with offspring
        population[elite] = offspring
        return population

    def evaluate_fitness(self, population):
        # Evaluate fitness of population
        fitness = np.zeros(len(population))
        for i, individual in enumerate(population):
            func_value = individual
            func_value = self.__call__(func_value)
            fitness[i] = func_value
        return fitness

# One-line description with the main idea
# Black Box Optimization using Genetic Algorithm with Adaptive Mutation Strategy
# A genetic algorithm that uses an adaptive mutation strategy to optimize black box functions with a population size of 100 individuals
# The algorithm evaluates the fitness of each individual in the population over 100 iterations and returns the best individual