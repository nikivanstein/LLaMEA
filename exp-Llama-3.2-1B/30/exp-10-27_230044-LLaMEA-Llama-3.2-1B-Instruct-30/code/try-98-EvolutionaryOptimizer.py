import numpy as np
import random
import math

class EvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.01
        self.selection_rate = 0.3

    def __call__(self, func):
        """
        Optimize the black box function using Evolutionary Optimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the population with random solutions
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

        # Run the evolutionary algorithm
        for _ in range(self.budget):
            # Select parents using selection
            parents = self.select_parents(self.population)

            # Perform crossover and mutation on parents
            offspring = self.crossover_and_mutate(parents)

            # Evaluate fitness of offspring
            fitness = [self.evaluate_fitness(offspring[i], func) for i in range(self.population_size)]

            # Select fittest offspring
            self.population = self.select_fittest(offspring, fitness)

            # Check if optimization is successful
            if np.allclose(self.population, self.population_size):
                return np.mean(self.population)

    def select_parents(self, population):
        # Select parents using tournament selection
        parents = []
        for _ in range(self.population_size):
            parent = random.choice(population)
            winner = np.max([self.evaluate_fitness(parent, func) for func in population])
            if winner!= 0:
                parents.append(parent)
        return parents

    def crossover_and_mutate(self, parents):
        # Perform crossover and mutation on parents
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                child = (parent1 + parent2) / 2
                child[0] = random.uniform(-5.0, 5.0)
                child[1] = random.uniform(-5.0, 5.0)
                child[2] = random.uniform(-5.0, 5.0)
                child[2] = random.uniform(-5.0, 5.0)
                offspring.append(child)
            else:
                offspring.append(parents[i])
        return offspring

    def evaluate_fitness(self, individual, func):
        """
        Evaluate the fitness of an individual.

        Args:
            individual (float): The individual to evaluate.
            func (function): The black box function to optimize.

        Returns:
            float: The fitness of the individual.
        """
        return func(individual)

    def select_fittest(self, offspring, fitness):
        # Select fittest offspring
        fittest = offspring[np.argmax(fitness)]
        return fittest

# Example usage:
func = lambda x: x**2
optimizer = EvolutionaryOptimizer(100, 2)
print(optimizer(func))