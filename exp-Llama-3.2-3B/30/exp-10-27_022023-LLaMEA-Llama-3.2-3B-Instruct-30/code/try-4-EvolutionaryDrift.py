import numpy as np
import random

class EvolutionaryDrift:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.refinement_prob = 0.3
        self.best_solution = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        if self.budget == 0:
            return self.best_solution

        for _ in range(self.budget):
            # Initialize population with random solutions
            population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

            # Evaluate population fitness
            fitness = [func(x) for x in population]

            # Update best solution if necessary
            if min(fitness) < self.best_fitness:
                self.best_solution = population[np.argmin(fitness)]
                self.best_fitness = min(fitness)

            # Apply evolutionary drift
            for _ in range(10):
                # Select parents using tournament selection
                parents = np.array([population[np.random.choice(range(self.population_size)), :] for _ in range(2)])

                # Apply crossover
                offspring = []
                for _ in range(self.population_size):
                    parent1, parent2 = parents[np.random.choice(range(2))]
                    child = (parent1 + parent2) * 0.5
                    child = np.clip(child, -5.0, 5.0)
                    offspring.append(child)

                # Introduce mutation with adaptive rate
                for i in range(self.population_size):
                    if random.random() < self.mutation_rate:
                        mutation = np.random.uniform(-1.0, 1.0, self.dim)
                        child = offspring[i] + mutation
                        child = np.clip(child, -5.0, 5.0)
                        offspring[i] = child

                # Refine selected individuals with probabilistic mutation
                refined_offspring = []
                for i in range(self.population_size):
                    if random.random() < self.refinement_prob:
                        mutation = np.random.uniform(-1.0, 1.0, self.dim)
                        child = offspring[i] + mutation
                        child = np.clip(child, -5.0, 5.0)
                        refined_offspring.append(child)
                    else:
                        refined_offspring.append(offspring[i])

                # Update population
                offspring = np.array(refined_offspring)

            # Update mutation rate
            self.mutation_rate *= 0.9

        return self.best_solution

# Example usage:
def func(x):
    return sum(x**2)

evolutionary_drift = EvolutionaryDrift(budget=100, dim=10)
best_solution = evolutionary_drift(func)
print(best_solution)