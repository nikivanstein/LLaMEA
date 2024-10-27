import numpy as np
import random
import time
import copy

class MultiSwarmEvolutionStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.crossover_probability = 0.9
        self.mutation_probability = 0.1
        self.swarm_size = 10
        self.fitness_history = []
        self.best_individuals = []

    def __call__(self, func):
        start_time = time.time()
        for i in range(self.budget):
            # Initialize swarms with random points
            swarms = [np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim)) for _ in range(self.swarm_size)]

            # Evaluate fitness for each swarm
            fitness = [np.array([func(individual) for individual in swarm]) for swarm in swarms]

            # Multi-swarm evolution
            for _ in range(100):  # 100 generations
                # Select swarms using tournament selection
                swarms = [swarm for _ in range(self.swarm_size)]
                for _ in range(self.population_size):
                    tournament = random.sample(range(self.swarm_size), 3)
                    swarms[tournament[np.argmin([np.min(individuals) for individuals in swarms[tournament]])]] += [swarm[tournament[np.argmin([np.min(individuals) for individuals in swarms[tournament]])]] for swarm in swarms[tournament]]

                # Crossover
                for j in range(self.swarm_size):
                    swarms[j] = [self.crossover(swarms[j][i], swarms[j][i+1]) for i in range(self.population_size-1)]

                # Mutation
                for j in range(self.swarm_size):
                    for i in range(self.population_size):
                        if random.random() < self.mutation_probability:
                            mutation = np.random.uniform(-1.0, 1.0, size=self.dim)
                            swarms[j][i] += mutation

                # Replace worst individual
                swarms = [np.sort(swarm, axis=0) for swarm in swarms]
                swarms = [swarm[np.argmin(fitness[j])] for j, swarm in enumerate(swarms)]

                # Evaluate fitness for each individual
                fitness = [np.array([func(individual) for individual in swarm]) for swarm in swarms]

            # Update fitness history and best individuals
            self.fitness_history.append(fitness)
            self.best_individuals.append(np.argmin([np.min(individuals) for individuals in swarms]))

        end_time = time.time()
        print(f"Optimization time: {end_time - start_time} seconds")
        return np.min([np.min(individuals) for individuals in swarms]), swarms[np.argmin([np.min(individuals) for individuals in swarms])]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_probability:
            child = (parent1 + parent2) / 2
            return child
        else:
            return parent1

# Example usage:
def func(x):
    return np.sum(x**2)

mses = MultiSwarmEvolutionStrategy(budget=100, dim=10)
best_fitness, best_swarm = mses(func)
print(f"Best fitness: {best_fitness}")
print(f"Best swarm: {best_swarm}")