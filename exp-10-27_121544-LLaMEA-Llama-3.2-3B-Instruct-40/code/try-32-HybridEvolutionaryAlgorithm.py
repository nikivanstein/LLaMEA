import numpy as np
import random
import copy

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.population_size = 20
        self.population = [copy.deepcopy(self.x0) for _ in range(self.population_size)]

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        for _ in range(self.budget):
            new_population = []
            for individual in self.population:
                # Crossover
                if random.random() < 0.4:
                    parent1, parent2 = random.sample(self.population, 2)
                    child = [random.uniform(parent1[i], parent2[i]) for i in range(self.dim)]
                else:
                    child = individual

                # Mutation
                if random.random() < 0.1:
                    mutation = random.uniform(-1, 1)
                    child = [child[i] + mutation if random.random() < 0.5 else child[i] - mutation for i in range(self.dim)]

                new_population.append(child)

            self.population = new_population

            # Evaluate fitness
            fitness = [func(individual) for individual in self.population]
            min_fitness = min(fitness)
            min_index = fitness.index(min_fitness)
            self.population = [individual for i, individual in enumerate(self.population) if i!= min_index]

            # Replace worst individual
            worst_individual = self.population[-1]
            self.population[-1] = copy.deepcopy(self.population[min_index])

        return self.population[0], func(self.population[0])

# Usage
budget = 100
dim = 10
func = lambda x: sum([i**2 for i in x])
algorithm = HybridEvolutionaryAlgorithm(budget, dim)
result = algorithm(func)
print(result)