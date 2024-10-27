import numpy as np
import random

class EvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.swarm_size = 10
        self.refinement_rate = 0.25

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        for _ in range(self.budget):
            fitness = np.array([func(x) for x in population])

            fittest = population[np.argsort(fitness)]

            offspring = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(fittest, 2)

                child = np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))

                if random.random() < self.crossover_rate:
                    child = self.crossover(child, parent1, parent2)

                if random.random() < self.mutation_rate:
                    child += np.random.uniform(-0.1, 0.1, self.dim)

                offspring.append(child)

            # Refine the population with probabilistic mutation
            refined_population = []
            for individual in population:
                if random.random() < self.refinement_rate:
                    refined_individual = self.mutate(individual)
                else:
                    refined_individual = individual
                refined_population.append(refined_individual)

            population = np.array(refined_population)

        # Return the best solution found
        return population[np.argmin(fitness)]

    def crossover(self, child, parent1, parent2):
        # Perform crossover between two parents
        child = np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))

        # Perform uniform crossover
        mask = np.random.choice([0, 1], size=self.dim, p=[0.5, 0.5])
        child[mask] = parent1[mask]

        return child

    def mutate(self, individual):
        # Perform uniform mutation
        mutation = np.random.uniform(-0.1, 0.1, self.dim)
        individual += mutation

        return individual

# Example usage
def func(x):
    return np.sum(x**2)

optimizer = EvolutionaryOptimizer(budget=100, dim=5)
best_solution = optimizer(func)
print("Best solution:", best_solution)