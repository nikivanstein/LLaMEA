import numpy as np
import random
import math

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.metaheuristic = Metaheuristic(budget, dim)

    def __call__(self, func):
        # Initialize the population
        population = [self.metaheuristic(func) for _ in range(100)]

        # Run the evolutionary algorithm
        for _ in range(100):
            # Select the fittest individuals
            fittest = sorted(population, key=lambda x: x.fitness, reverse=True)[:self.metaheuristic.budget]

            # Select parents using tournament selection
            parents = [fittest[i] for i in random.sample(range(self.metaheuristic.budget), len(fittest))]

            # Crossover (recombination)
            offspring = []
            for _ in range(self.metaheuristic.budget):
                parent1, parent2 = random.sample(parents, 2)
                child = (parent1 + parent2) / 2
                offspring.append(child)

            # Mutate the offspring
            for i in range(self.metaheuristic.budget):
                if random.random() < 0.45:
                    offspring[i] = random.uniform(-5.0, 5.0)

            # Replace the least fit individuals with the new offspring
            population = [x for x in fittest if x.fitness > x.fitness.max() - 0.1] + [x for x in offspring if x.fitness > x.fitness.max() - 0.1]

        return population[0]

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# This algorithm uses a combination of mutation and crossover to evolve a population of individuals, each representing a potential solution to the black box optimization problem.
# The algorithm selects the fittest individuals and uses tournament selection, crossover, and mutation to generate a new population, which is then replaced with the least fit individuals.
# The process is repeated for a fixed number of generations, with the fittest individuals at the end of each generation being selected as the next population.