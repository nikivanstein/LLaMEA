import numpy as np
import random
import math
import copy

class GeneticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitness_scores = []

    def __call__(self, func, population_size=100, mutation_rate=0.01, selection_rate=0.3, crossover_rate=0.7):
        """
        Optimize the black box function using Genetic Optimizer.

        Args:
            func (function): The black box function to optimize.
            population_size (int): The size of the population.
            mutation_rate (float): The probability of mutation.
            selection_rate (float): The probability of selection.
            crossover_rate (float): The probability of crossover.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize population with random individuals
        for _ in range(population_size):
            individual = self.generate_individual(func, self.dim)
            self.population.append(individual)

        # Evaluate the population
        for _ in range(self.budget):
            # Select parents using selection rate
            parents = self.select_parents(population_size, selection_rate)
            # Crossover parents to create offspring
            offspring = self.crossover(parents, crossover_rate)
            # Mutate offspring using mutation rate
            offspring = self.mutate(offspring, mutation_rate)
            # Replace parents with offspring
            self.population = self.population[:population_size] + offspring

        # Return the fittest individual
        return self.population[0]

    def generate_individual(self, func, dim):
        """
        Generate a random individual using a genetic algorithm.

        Args:
            func (function): The black box function to optimize.
            dim (int): The dimensionality.

        Returns:
            list: The generated individual.
        """
        individual = [random.uniform(-5.0, 5.0) for _ in range(dim)]
        return individual

    def select_parents(self, population_size, selection_rate):
        """
        Select parents using selection rate.

        Args:
            population_size (int): The size of the population.
            selection_rate (float): The probability of selection.

        Returns:
            list: The selected parents.
        """
        parents = []
        for _ in range(population_size):
            fitness = self.fitness(func, individual) / 100  # Normalize fitness
            if random.random() < selection_rate:
                parents.append((individual, fitness))
        return parents

    def crossover(self, parents, crossover_rate):
        """
        Crossover parents to create offspring.

        Args:
            parents (list): The parents.
            crossover_rate (float): The probability of crossover.

        Returns:
            list: The offspring.
        """
        offspring = []
        while len(offspring) < parents[0][0].len():
            parent1, parent2 = random.sample(parents, 2)
            child = (parent1[:], parent2[:])
            if random.random() < crossover_rate:
                child[0], child[1] = child[1], child[0]
            offspring.append(child)
        return offspring

    def mutate(self, offspring, mutation_rate):
        """
        Mutate offspring using mutation rate.

        Args:
            offspring (list): The offspring.
            mutation_rate (float): The probability of mutation.

        Returns:
            list: The mutated offspring.
        """
        mutated_offspring = []
        for individual in offspring:
            mutated_individual = copy.deepcopy(individual)
            for i in range(len(individual)):
                if random.random() < mutation_rate:
                    mutated_individual[i] += random.uniform(-1, 1)
            mutated_offspring.append(mutated_individual)
        return mutated_offspring

    def fitness(self, func, individual):
        """
        Evaluate the fitness of an individual.

        Args:
            func (function): The black box function.
            individual (list): The individual.

        Returns:
            float: The fitness of the individual.
        """
        return func(individual)

# Test the optimizer
def test_optimizer(func, budget=1000):
    optimizer = GeneticOptimizer(budget)
    return optimizer(func, selection_rate=0.3, crossover_rate=0.7)

# Test the optimizer on the BBOB test suite
def test_bbob(func):
    # Define the BBOB test suite
    test_suite = {
        "noiseless functions": [
            {"name": "tanh", "func": np.tanh},
            {"name": "sin", "func": np.sin},
            {"name": "cos", "func": np.cos},
            {"name": "exp", "func": np.exp}
        ]
    }

    # Evaluate the optimizer on the test suite
    for func_name, func_func in test_suite.items():
        print(f"Optimizing {func_name}...")
        optimized_value = test_optimizer(func_func, budget=1000)
        print(f"Optimized value: {optimized_value}")
        print(f"Fitness: {func(optimized_value)}")
        print()

# Run the test
test_bbob(func)