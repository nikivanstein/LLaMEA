import numpy as np
import random
import math

class GeneticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = None
        self.population_history = None

    def __call__(self, func):
        """
        Optimize the black box function using Genetic Optimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize population with random solutions
        self.population = self.generate_population(func, self.budget, self.dim)

        # Run the optimization algorithm
        for _ in range(self.budget):
            # Evaluate fitness of each individual
            fitnesses = [self.evaluate_fitness(individual, func) for individual in self.population]

            # Select parents using tournament selection
            parents = self.select_parents(fitnesses)

            # Crossover (recombination) offspring
            offspring = self.crossover(parents)

            # Mutate offspring
            mutated_offspring = self.mutate(offspring)

            # Replace least fit individuals with new offspring
            self.population = self.population[:self.population_size - len(mutated_offspring)]
            self.population.extend(mutated_offspring)

            # Replace least fit individual with the fittest individual
            self.population_history.append(self.population[fitnesses.index(max(fitnesses))])

        # Return the fittest individual
        return self.population[fitnesses.index(max(fitnesses))]

    def generate_population(self, func, budget, dim):
        """
        Generate a population of random solutions.

        Args:
            func (function): The black box function to optimize.
            budget (int): The number of function evaluations.
            dim (int): The dimensionality of the problem.

        Returns:
            list: A list of random solutions.
        """
        return [np.random.rand(dim) for _ in range(budget)]

    def select_parents(self, fitnesses):
        """
        Select parents using tournament selection.

        Args:
            fitnesses (list): A list of fitness values.

        Returns:
            list: A list of parent solutions.
        """
        parents = []
        for _ in range(len(fitnesses)):
            tournament_size = random.randint(1, len(fitnesses))
            tournament = random.choices(fitnesses, weights=[fitness / len(fitnesses) for fitness in fitnesses], k=tournament_size)
            tournament = [individual for individual in tournament if individual in fitnesses]
            parents.append(tournament[0])
        return parents

    def crossover(self, parents):
        """
        Crossover (recombination) offspring.

        Args:
            parents (list): A list of parent solutions.

        Returns:
            list: A list of offspring solutions.
        """
        offspring = []
        while len(offspring) < len(parents):
            parent1, parent2 = random.sample(parents, 2)
            offspring.append(self.crossover_helper(parent1, parent2))
        return offspring

    def crossover_helper(self, parent1, parent2):
        """
        Crossover (recombination) offspring.

        Args:
            parent1 (float): The first parent solution.
            parent2 (float): The second parent solution.

        Returns:
            float: The offspring solution.
        """
        crossover_point = random.randint(1, self.dim - 1)
        return parent1[:crossover_point] + parent2[crossover_point:]

    def mutate(self, offspring):
        """
        Mutate offspring.

        Args:
            offspring (list): A list of offspring solutions.

        Returns:
            list: A list of mutated offspring solutions.
        """
        mutated_offspring = []
        for individual in offspring:
            mutated_individual = individual.copy()
            mutated_individual[random.randint(0, self.dim - 1)] += random.uniform(-0.1, 0.1)
            mutated_offspring.append(mutated_individual)
        return mutated_offspring

    def evaluate_fitness(self, individual, func):
        """
        Evaluate the fitness of an individual.

        Args:
            individual (float): The individual solution.
            func (function): The black box function to optimize.

        Returns:
            float: The fitness value.
        """
        return func(individual)

# Description: Black Box Optimization using Genetic Algorithm with Neural Network Evolutionary Strategies
# Code: 