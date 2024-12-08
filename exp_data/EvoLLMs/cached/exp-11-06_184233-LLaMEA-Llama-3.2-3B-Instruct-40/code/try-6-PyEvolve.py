import numpy as np
from deap import base, creator, tools, algorithms
from scipy import optimize

class PyEvolve:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_prob = 0.1
        self.crossover_prob = 0.5
        self.elitism = 10

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.population = tools.initRepeat(list, creator.Individual, np.random.uniform(-5.0, 5.0, self.dim), self.population_size)
        self.hall_of_fame = tools.HallOfFame(self.elitism)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.select = tools.selTournament

    def __call__(self, func):
        for _ in range(self.budget):
            offspring = algorithms.varAnd(self.population, self.mutation_prob, self.crossover_prob)
            fits = self.stats.compile(offspring)
            self.population = tools.selBest(offspring, self.elitism + fits.index(min(fits)))
            self.hall_of_fame.update(self.population)
            f_best = self.population[0].fitness.values[0]
            x_best = self.population[0].individual
            print(f"Best found so far at generation {_+1}: f({x_best}) = {f_best}")

        # Refine the strategy of the selected solution
        best_individual = self.population[0]
        best_fitness = best_individual.fitness.values[0]
        x_best = best_individual.individual
        print(f"Best found so far: f({x_best}) = {best_fitness}")

        # Use a more efficient optimization method (e.g., gradient-based optimization)
        res = optimize.minimize(func, x_best, method="SLSQP")
        print(f"Refined best found: f({res.x}) = {res.fun}")

        return res.x

# Example usage:
func = lambda x: x[0]**2 + x[1]**2
pyevolve = PyEvolve(100, 2)
x_best = pyevolve(func)
print(f"Final best found: f({x_best}) = {func(x_best)}")