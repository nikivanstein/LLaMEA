# Description: Evolutionary Black Box Optimization with Refining Strategy
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import differential_evolution

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

def bbob_optimize(func, budget, dim):
    optimizer = EvolutionaryBlackBoxOptimizer(budget, dim)
    best_individual = optimizer(population)
    best_fitness = fitness(best_individual)
    best_individual = best_individual.tolist()
    return best_individual, best_fitness

def fitness(individual):
    return func(individual)

def evaluate_fitness(individual, func):
    return func(individual)

# Example usage:
def func(x):
    return sum(x)

best_individual, best_fitness = bbob_optimize(func, 100, 10)
print("Best Individual:", best_individual)
print("Best Fitness:", best_fitness)