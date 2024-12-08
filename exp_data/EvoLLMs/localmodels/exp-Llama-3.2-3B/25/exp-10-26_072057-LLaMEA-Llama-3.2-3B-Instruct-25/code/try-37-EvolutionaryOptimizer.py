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
        self.refinement_probability = 0.25

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
                    child = np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))

                if random.random() < self.refinement_probability:
                    # Refine the child by changing individual lines
                    child = self.refine_individual(child)

                offspring.append(child)

            population = np.array(offspring)

        best_solution = population[np.argmin(fitness)]
        return best_solution

    def refine_individual(self, individual):
        refined_individual = individual.copy()
        for i in range(self.dim):
            if random.random() < self.refinement_probability:
                refined_individual[i] += np.random.uniform(-0.1, 0.1)
        return refined_individual

# Example usage
def func(x):
    return np.sum(x**2)

optimizer = EvolutionaryOptimizer(budget=100, dim=5)
best_solution = optimizer(func)
print("Best solution:", best_solution)