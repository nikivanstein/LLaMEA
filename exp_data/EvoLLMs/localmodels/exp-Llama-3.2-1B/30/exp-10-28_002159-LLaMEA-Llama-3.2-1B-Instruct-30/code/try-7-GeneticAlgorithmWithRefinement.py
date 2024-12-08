import random
import numpy as np

class GeneticAlgorithmWithRefinement:
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

        for _ in range(self.budget):
            for i, individual in enumerate(self.population):
                fitness_scores[i] = fitness(individual)
            best_individual = self.population[np.argmax(fitness_scores)]
            new_individual = [random.uniform(self.search_spaces[j][0], self.search_spaces[j][1]) for j in range(self.dim)]
            new_individual = np.array(new_individual)
            if fitness(individual) > fitness(best_individual):
                self.population[i] = new_individual
                self.fitness_scores[i] = fitness(individual)

        return self.population

    def mutate(self, individual):
        if random.random() < 0.3:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate(self, func):
        return func(np.array(self.population))

    def select(self, fitness_scores, num_parents):
        fitness_indices = np.argsort(fitness_scores)[::-1][:num_parents]
        return [self.population[i] for i in fitness_indices]

    def crossover(self, parent1, parent2):
        if random.random() < 0.5:
            index = random.randint(0, self.dim - 1)
            child1 = np.copy(parent1)
            child1[index] = parent2[index]
            return child1
        else:
            return parent2

    def mutate_crossover(self, individual1, individual2):
        if random.random() < 0.3:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual1, individual2

    def evolve(self):
        population = self.population
        while True:
            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent1, parent2)
                new_individual1, new_individual2 = self.mutate_crossover(child1, child2)
                new_population.extend([new_individual1, new_individual2])
            population = new_population
            fitness_scores = np.zeros((len(population), self.dim))
            for i, individual in enumerate(population):
                fitness_scores[i] = fitness(individual)
            best_individual = population[np.argmax(fitness_scores)]
            new_population = self.select(fitness_scores, self.population_size // 2)
            if fitness(best_individual) > fitness(self.population[np.argmax(fitness_scores)]):
                population = new_population
            return population

# Example usage
ga = GeneticAlgorithmWithRefinement(budget=100, dim=5)
func = lambda x: np.sin(x)
best_solution = ga.evolve()
print("Best solution:", best_solution)
print("Best fitness:", ga.evaluate(func))