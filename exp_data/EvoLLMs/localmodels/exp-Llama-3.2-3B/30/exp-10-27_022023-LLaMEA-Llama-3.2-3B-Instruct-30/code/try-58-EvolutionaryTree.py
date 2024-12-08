import numpy as np
import random

class EvolutionaryTree:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.tree_depth = 5
        self.refine_probability = 0.3
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

            # Apply evolutionary tree evolution
            for _ in range(self.tree_depth):
                # Select parents using tournament selection
                parents = np.array([population[np.random.choice(range(self.population_size)), :] for _ in range(2)])

                # Apply crossover and mutation
                offspring = []
                for _ in range(self.population_size):
                    parent1, parent2 = parents[np.random.choice(range(2))]
                    child = (parent1 + parent2) * 0.5 + np.random.uniform(-0.5, 0.5, self.dim)
                    child = np.clip(child, -5.0, 5.0)
                    offspring.append(child)

                # Update population
                population = np.array(offspring)

            # Refine the best solution with probabilistic refinement
            if np.random.rand() < self.refine_probability:
                new_individual = population[np.random.choice(range(self.population_size))]
                new_fitness = func(new_individual)
                if new_fitness < self.best_fitness:
                    self.best_solution = new_individual
                    self.best_fitness = new_fitness

        return self.best_solution

# Example usage:
def func(x):
    return sum(x**2)

evolutionary_tree = EvolutionaryTree(budget=100, dim=10)
best_solution = evolutionary_tree(func)
print(best_solution)