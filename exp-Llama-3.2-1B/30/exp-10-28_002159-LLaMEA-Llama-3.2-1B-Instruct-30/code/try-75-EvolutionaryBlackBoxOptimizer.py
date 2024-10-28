import numpy as np
import random

class EvolutionaryBlackBoxOptimizer:
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

        def mutate(individual):
            if random.random() < 0.3:
                index = random.randint(0, self.dim - 1)
                self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
            return individual

        def evaluate(func, population):
            return np.array([func(individual) for individual in population])

        def selection(population, fitness_scores):
            return np.random.choice(len(population), size=self.population_size, p=fitness_scores / np.sum(fitness_scores))

        def crossover(parent1, parent2):
            if random.random() < 0.3:
                index = random.randint(0, self.dim - 1)
                child = np.concatenate((parent1[:index], parent2[index:]))
            else:
                child = np.concatenate((parent1, parent2))
            return child

        def mutation_fitness(individual):
            if random.random() < 0.3:
                index = random.randint(0, self.dim - 1)
                self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
            return individual

        def select_parents(population, fitness_scores):
            return selection(population, fitness_scores)

        def crossover_with_parents(parent1, parent2):
            return crossover(parent1, parent2)

        def mutate_with_parents(individual):
            return mutate(individual)

        # Main loop
        for _ in range(self.budget):
            new_population = []
            for _ in range(self.population_size):
                parent1 = select_parents(self.population, self.fitness_scores)
                parent2 = select_parents(self.population, self.fitness_scores)
                child = crossover_with_parents(parent1, parent2)
                child = mutate_with_parents(child)
                new_population.append(child)
            self.population = new_population

        # Evaluate the new population
        fitness_scores = evaluate(func, self.population)
        best_individual = np.argmax(fitness_scores)
        best_individual = self.population[best_individual]
        new_individual = mutation_fitness(best_individual)
        return new_individual

    def evaluate(self, func):
        return func(np.array(self.population))

# Example usage
def func(x):
    return np.sum(x**2)

optimizer = EvolutionaryBlackBoxOptimizer(100, 10)
best_individual = optimizer(__call__, func)
print("Best individual:", best_individual)
print("Best fitness:", np.sum(best_individual**2))