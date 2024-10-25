import random
import numpy as np

class MetaHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]  # Define the search space
        self.best_func = None  # Initialize the best function found so far
        self.best_fitness = float('inf')  # Initialize the best fitness found so far
        self.iterations = 0  # Initialize the number of iterations
        self.population = []  # Initialize the population

    def __call__(self, func, max_evals):
        # Evaluate the function up to max_evals times
        for _ in range(max_evals):
            # Randomly sample a point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            fitness = func(point)
            # If the fitness is better than the current best fitness, update the best function and fitness
            if fitness < self.best_fitness:
                self.best_func = func
                self.best_fitness = fitness
            # If the fitness is equal to the current best fitness, update the best function if it has a lower budget
            elif fitness == self.best_fitness and self.budget < self.best_func.budget:
                self.best_func = func
                self.best_fitness = fitness
            # Add the new individual to the population
            self.population.append(point)
        # Return the best function found
        return self.best_func

    def set_budget(self, budget):
        self.budget = budget

    def get_best_func(self):
        return self.best_func

    def mutate(self, individual):
        # Refine the strategy by changing the individual lines
        for _ in range(5):
            i = random.randint(0, self.dim - 1)
            j = random.randint(0, self.dim - 1)
            self.population[i], self.population[j] = self.population[j], self.population[i]
        # Return the mutated individual
        return self.population

    def crossover(self, parent1, parent2):
        # Combine the parents to create a new individual
        child = (parent1[0], parent2[0])
        for i in range(1, self.dim):
            child += (parent1[i] + parent2[i]) / 2
        # Return the new individual
        return child

    def evolve(self):
        # Evolve the population using mutation and crossover
        new_population = []
        while len(self.population) > 0:
            # Select two parents
            parent1 = random.choice(self.population)
            parent2 = random.choice(self.population)
            # Create a new individual by crossover and mutation
            child = self.crossover(parent1, parent2)
            # Add the child to the new population
            new_population.append(child)
        # Replace the old population with the new population
        self.population = new_population
        # Update the best function and fitness
        best_func = self.__call__(test_func, 100)
        best_fitness = best_func.budget
        # Print the best function found
        print("Best function:", best_func)
        print("Best fitness:", best_fitness)

# Description: "MetaHeuristics for Black Box Optimization"
# Code: 
# ```python
meta_heuristic = MetaHeuristic(100, 10)
meta_heuristic.evolve()