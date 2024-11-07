import numpy as np
from scipy.optimize import differential_evolution
from deap import base, creator, tools, algorithms

class LeverageExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.x_best = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
        self.f_best = np.inf
        self.elite_size = int(self.budget * 0.2)

    def __call__(self, func):
        population = tools.initRandomIndividual(self.dim, self.lower_bound, self.upper_bound)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

        self.population = tools.initRepeat(population, creator.Individual, self.elite_size)

        for _ in range(self.budget - self.elite_size):
            # Selection
            offspring = tools.select(self.population, len(self.population) - self.elite_size, tournsize=3)

            # Crossover
            offspring = algorithms.varAnd(offspring, self.dim, self.lower_bound, self.upper_bound)

            # Mutation
            offspring = algorithms.mutShake(offspring, indpb=0.1)

            # Evaluate
            for kid in offspring:
                f_kid = func(kid.x)
                if f_kid < kid.fitness.values[0]:
                    self.population[kid] = kid

            # Local search
            x_local = self.x_best
            f_local = func(x_local)

            # Leverage the best exploration point
            if f_local < self.f_best:
                self.x_best = x_local
                self.f_best = f_local

            # Differential evolution for local search
            x_dev = differential_evolution(func, [(self.lower_bound, self.upper_bound) for _ in range(self.dim)])
            f_dev = func(x_dev.x)

            # Update the best point if the local search is better
            if f_dev < self.f_best:
                self.x_best = x_dev.x
                self.f_best = f_dev

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

le = LeverageExploration(budget=100, dim=2)
le(func)
print("Best point:", le.x_best)
print("Best function value:", le.f_best)
