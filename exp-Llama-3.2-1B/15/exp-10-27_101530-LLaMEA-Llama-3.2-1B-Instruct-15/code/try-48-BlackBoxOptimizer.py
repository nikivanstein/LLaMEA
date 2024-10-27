import random
import numpy as np
from collections import deque

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.population_size = 100
        self.population_deletion_rate = 0.05
        self.population_size_decrease_rate = 0.01
        self.population_deletion_interval = 1000
        self.population_deletion_counter = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def mutate(self, individual):
        # Randomly swap two elements in the individual
        i, j = random.sample(range(self.dim), 2)
        individual[i], individual[j] = individual[j], individual[i]
        return individual

    def crossover(self, parent1, parent2):
        # Select a random crossover point
        crossover_point = random.randint(1, self.dim - 1)
        # Create a child individual by combining the parents
        child = [parent1[i] for i in range(crossover_point)] + [parent2[i] for i in range(crossover_point, self.dim)]
        return child

    def selection(self, population):
        # Select the fittest individuals
        fittest_individuals = sorted(population, key=self.evaluate_fitness, reverse=True)[:self.population_size]
        return fittest_individuals

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual
        func_value = self.func(individual)
        return func_value

    def update_population(self, population):
        # Update the population with new individuals
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(population, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        population = new_population

    def update_population_size(self, new_population_size):
        # Update the population size
        self.population_size = new_population_size
        # Remove old individuals
        self.population_deletion_counter += 1
        while self.population_deletion_counter > self.population_deletion_interval:
            old_individual = random.choice(self.population)
            if self.evaluate_fitness(old_individual) >= self.evaluate_fitness(self.search_space[0], self.search_space[1]):
                self.population.remove(old_individual)
            else:
                self.population_deletion_counter -= 1
        # Add new individuals
        self.population = new_population

    def run(self, budget):
        # Run the optimization algorithm
        population = deque([self.search_space])
        while self.func_evaluations < budget:
            self.update_population(population)
            self.update_population_size(self.population_size)
            self.run_single(population)
        return self.search_space

    def run_single(self, population):
        # Run a single iteration of the algorithm
        while len(population) > 0:
            individual = population.popleft()
            fitness = self.evaluate_fitness(individual)
            if fitness >= self.evaluate_fitness(self.search_space[0], self.search_space[1]):
                return individual
            else:
                self.population_deletion_counter += 1
                while self.population_deletion_counter > self.population_deletion_interval:
                    old_individual = random.choice(self.population)
                    if self.evaluate_fitness(old_individual) >= self.evaluate_fitness(self.search_space[0], self.search_space[1]):
                        self.population.remove(old_individual)
                    else:
                        self.population_deletion_counter -= 1
                self.population.append(individual)