import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.search_spaces[i][0], self.search_spaces[i][1]) for i in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        def fitness(individual):
            return func(individual)

        def fitness_better(individual):
            return fitness(individual) > fitness(individual)

        for _ in range(self.budget):
            for i, individual in enumerate(self.population):
                fitness_scores_i = fitness(individual)
                if fitness_better(individual):
                    fitness_scores_i[fitness_scores_i.index(max(fitness_scores_i))] = fitness_scores_i
                    self.population[i] = individual
            best_individual = self.population[np.argmax(fitness_scores_i)]
            new_individual = [random.uniform(self.search_spaces[j][0], self.search_spaces[j][1]) for j in range(self.dim)]
            new_individual = np.array(new_individual)
            if fitness_better(new_individual):
                self.population[i] = new_individual

        return self.population

    def mutate(self, individual):
        if random.random() < 0.3:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate(self, func):
        return func(np.array(self.population))

class EvolutionaryBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim
        self.population = self.population

    def __call__(self, func):
        def fitness(individual):
            return func(individual)

        def fitness_better(individual):
            return fitness(individual) > fitness(individual)

        for _ in range(self.budget):
            for i, individual in enumerate(self.population):
                fitness_scores_i = fitness(individual)
                if fitness_better(individual):
                    fitness_scores_i[fitness_scores_i.index(max(fitness_scores_i))] = fitness_scores_i
                    self.population[i] = individual
            best_individual = self.population[np.argmax(fitness_scores_i)]
            new_individual = [random.uniform(self.search_spaces[j][0], self.search_spaces[j][1]) for j in range(self.dim)]
            new_individual = np.array(new_individual)
            if fitness_better(new_individual):
                self.population[i] = new_individual

        return self.population

    def mutate(self, individual):
        if random.random() < 0.3:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate(self, func):
        return func(np.array(self.population))

# Example usage:
optimizer = EvolutionaryBlackBoxOptimizer(100, 10)
func = lambda x: np.sin(x)
best_individual = optimizer(1000)
print(best_individual)

# The selected solution to update is:
# "Evolutionary Black Box Optimization"