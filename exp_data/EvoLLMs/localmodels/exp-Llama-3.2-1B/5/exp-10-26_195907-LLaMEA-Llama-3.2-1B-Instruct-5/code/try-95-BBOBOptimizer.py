# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        # Initialize population with random solutions
        population = [self.evaluate_fitness(x) for x in self.search_space]

        # Evolve population for the specified number of generations
        for _ in range(100):  # Increase the number of generations to improve convergence
            # Select parents using tournament selection
            parents = self.tournament_selection(population)

            # Crossover (reproduce) parents to create offspring
            offspring = self.crossover(parents)

            # Mutate offspring to introduce diversity
            offspring = self.mutate(offspring)

            # Replace worst individual in population with the best offspring
            worst_index = np.argmin(population)
            population[worst_index] = offspring[worst_index]

            # Update search space with the new individual
            self.search_space = np.vstack((self.search_space, offspring[-1]))

            # Limit population size to the budget
            population = population[:self.budget]

        # Return the fittest individual as the best solution
        return self.evaluate_fitness(population[0])

    def tournament_selection(self, population):
        # Select the fittest individual from each subset of population
        selected_individuals = []
        for _ in range(len(population) // 2):
            subset = random.sample(population, len(population) // 2)
            individual = min(subset, key=self.evaluate_fitness)
            selected_individuals.append(individual)

        return selected_individuals

    def crossover(self, parents):
        # Perform crossover to create offspring
        offspring = []
        while len(offspring) < len(parents):
            parent1, parent2 = random.sample(parents, 2)
            if np.random.rand() < 0.5:
                offspring.append(parent1)
            else:
                offspring.append(parent2)

        return offspring

    def mutate(self, offspring):
        # Mutate offspring to introduce diversity
        mutated_offspring = []
        for individual in offspring:
            if np.random.rand() < 0.05:
                individual = random.uniform(self.search_space[0], self.search_space[1])
            mutated_offspring.append(individual)

        return mutated_offspring

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# A novel metaheuristic algorithm that uses tournament selection, crossover, and mutation to optimize black box functions.