# Description: Evolutionary Black Box Optimization Algorithm
# Code: 
# ```python
import random
import numpy as np

class EvolutionaryBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim
        self.best_individual = None

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.search_spaces[i][0], self.search_spaces[i][1]) for i in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        def fitness(individual):
            return func(individual)

        def fitness_scores(individual):
            return np.array([fitness(individual)])

        def mutate(individual):
            if random.random() < 0.3:
                index = random.randint(0, self.dim - 1)
                self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
            return individual

        for _ in range(self.budget):
            for i, individual in enumerate(self.population):
                fitness_scores[i] = fitness_scores(i)
                best_individual = self.population[np.argmax(fitness_scores)]
                new_individual = mutate(individual)
                self.population[i] = new_individual
                fitness_scores[i] = fitness_scores(i)

        self.best_individual = self.population[np.argmax(fitness_scores)]

        return self.population

    def evaluate(self, func):
        return func(self.best_individual)

# Trace: 
# Description: Evolutionary Black Box Optimization Algorithm
# Code: 
# ```python
# ```python
def fitness_bbb(func, individual):
    return func(individual)

def mutate_bbb(individual):
    return individual + [random.uniform(-1, 1) for _ in range(len(individual))]

def bbboptimizer(budget, dim):
    return EvolutionaryBlackBoxOptimizer(budget, dim)

# Test the algorithm
func = lambda x: x**2
individual = bbboptimizer(100, 10)
best_individual = bbboptimizer(100, 10).evaluate(func)
print("Best individual:", best_individual)
print("Fitness:", bbboptimizer(100, 10).evaluate(func))