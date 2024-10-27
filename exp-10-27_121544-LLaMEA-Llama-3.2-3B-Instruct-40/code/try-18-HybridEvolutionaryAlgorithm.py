import numpy as np
from scipy.optimize import differential_evolution
from deap import base, creator, tools, algorithms
import random

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        population_size = 100
        mutation_rate = 0.1
        cxpb = 0.4

        # Initialize the population
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, self.bounds[0][0], self.bounds[0][1])
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, self.dim)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", func)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=self.bounds[0][0], up=self.bounds[0][1], eta=20.0)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Initialize the population
        population = toolbox.population(n=population_size)
        HGA = algorithms.HGA(toolbox, ngen=100, cxpb=cxpb, mutpb=mutation_rate, stats=tools.Statistics(lambda ind: ind.fitness.values))
        HGA evolve(population)

        # Evaluate the fitness of the population
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Select the best individual
        best_individual = tools.selBest(population, 1)[0]

        return best_individual.f, best_individual

# Usage
def func(individual):
    return np.sum(np.abs(np.sin(np.cos(np.array(individual)))) ** 2)

heacombbo = HybridEvolutionaryAlgorithm(50, 2)
best_individual = heacombbo(func)
print(best_individual)