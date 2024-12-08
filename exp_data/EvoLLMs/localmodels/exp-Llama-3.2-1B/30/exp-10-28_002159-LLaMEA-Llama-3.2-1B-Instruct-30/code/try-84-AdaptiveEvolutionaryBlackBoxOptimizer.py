import random
import numpy as np

class AdaptiveEvolutionaryBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim
        self.best_individual = None
        self.best_fitness = np.inf

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
            new_individual = self.evaluate_fitness(best_individual)
            if fitness(new_individual) > self.best_fitness:
                self.population[i] = new_individual
                self.fitness_scores[i] = fitness(new_individual)
                if fitness(new_individual) > self.best_fitness:
                    self.best_individual = new_individual
                    self.best_fitness = fitness(new_individual)

        return self.population

    def evaluate_fitness(self, individual):
        updated_individual = self.evaluate(individual)
        if random.random() < 0.3:
            updated_individual = self.mutate(updated_individual)
        return updated_individual

    def mutate(self, individual):
        if random.random() < 0.01:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate(self, func):
        return func(np.array(self.population))

# Example usage:
optimizer = AdaptiveEvolutionaryBlackBoxOptimizer(budget=100, dim=10)
func = lambda x: x**2
optimizer.population = optimizer.init_population()
optimizer.optimize(func)
print(optimizer.population)