import random
import numpy as np

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
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Initialize the population with random individuals
        population = self.initialize_population(func, self.budget, self.dim)

        # Evaluate the population for a fixed number of times
        num_evaluations = 10
        fitness_values = [self.evaluate_fitness(individual, func) for individual in population]

        # Select the fittest individuals
        fittest_individuals = [individual for individual, fitness in zip(population, fitness_values) if fitness > 0.5]

        # Refine the strategy by changing the mutation probability
        mutation_probability = 0.05
        for _ in range(100):
            # Select two random individuals
            parent1, parent2 = random.sample(fittest_individuals, 2)

            # Create a new individual by combining the parents
            child = self.crossover(parent1, parent2)

            # Mutate the child
            if random.random() < mutation_probability:
                child = self.mutate(child)

            # Evaluate the child
            fitness = self.evaluate_fitness(child, func)

            # Update the fittest individuals
            fittest_individuals.append((child, fitness))

        # Replace the old population with the new one
        population = fittest_individuals[:self.budget]

        return population

    def initialize_population(self, func, budget, dim):
        # Initialize the population with random individuals
        population = []
        for _ in range(budget):
            individual = [random.uniform(-5.0, 5.0) for _ in range(dim)]
            population.append(individual)
        return population

    def evaluate_fitness(self, individual, func):
        # Evaluate the individual
        fitness = func(individual)
        return fitness

    def crossover(self, parent1, parent2):
        # Crossover the parents to create a new individual
        child = parent1[:len(parent1)//2] + parent2[len(parent2)//2:]
        return child

    def mutate(self, individual):
        # Mutate the individual by changing a random element
        mutated_individual = individual.copy()
        for i in range(len(individual)):
            if random.random() < 0.1:
                mutated_individual[i] += random.uniform(-0.1, 0.1)
        return mutated_individual

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 