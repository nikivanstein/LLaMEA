import numpy as np
import random
import time
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.population_size = 20
        self.mutation_rate = 0.1
        self.crossover_rate = 0.4
        self.fitness_function = None

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        if self.fitness_function is None:
            self.fitness_function = func

        for _ in range(self.population_size):
            # Select parents
            parents = random.sample(range(self.population_size), 2)

            # Crossover
            child1, child2 = self.crossover(parents)

            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            # Evaluate fitness
            fitness1 = self.evaluate_fitness(child1)
            fitness2 = self.evaluate_fitness(child2)

            # Replace worst individual
            self.replace_worst(child1, fitness1)
            self.replace_worst(child2, fitness2)

        # Refine strategy
        self.refine_strategy()

        # Return best individual
        return self.get_best_individual()

    def crossover(self, parents):
        child1 = parents[0].copy()
        child2 = parents[1].copy()

        # Crossover probability
        if random.random() < self.crossover_rate:
            for i in range(self.dim):
                if random.random() < 0.5:
                    child1[i] = parents[0][i]
                else:
                    child2[i] = parents[1][i]

        return child1, child2

    def mutate(self, individual):
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                individual[i] += random.uniform(-1, 1)

        return individual

    def evaluate_fitness(self, individual):
        return self.fitness_function(individual)

    def replace_worst(self, individual, fitness):
        if fitness < self.get_best_individual()[1]:
            self.population.remove(individual)
            self.population.append(individual)

    def refine_strategy(self):
        # Select 20% of the population
        selected_individuals = random.sample(self.population, int(0.2 * self.population_size))

        # Change strategy for selected individuals
        for individual in selected_individuals:
            for i in range(self.dim):
                if random.random() < 0.4:
                    individual[i] += random.uniform(-1, 1)

    def get_best_individual(self):
        return min(self.population, key=lambda x: self.evaluate_fitness(x))

# Example usage:
if __name__ == "__main__":
    budget = 100
    dim = 10
    func = lambda x: x[0]**2 + x[1]**2
    heacombbo = HybridEvolutionaryAlgorithm(budget, dim)
    start_time = time.time()
    best_individual, best_fitness = heacombbo(func)
    print("Best individual:", best_individual)
    print("Best fitness:", best_fitness)
    print("Time taken:", time.time() - start_time)