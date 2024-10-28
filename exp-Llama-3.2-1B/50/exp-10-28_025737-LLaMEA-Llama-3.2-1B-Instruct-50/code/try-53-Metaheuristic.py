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

class Mutation:
    def __init__(self, mutation_rate, dim):
        self.mutation_rate = mutation_rate
        self.dim = dim
        self.new_individual = []

    def mutate(self, individual):
        # Refine the strategy by changing the individual lines
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                # Change the value of the individual
                new_value = random.uniform(-5.0, 5.0)
                self.new_individual.append([x + new_value for x in individual[i]])

        return self.new_individual

class Selection:
    def __init__(self, num_individuals, dim):
        self.num_individuals = num_individuals
        self.dim = dim
        self.fitness_values = []

    def select(self, individual):
        # Select the best individual based on the fitness values
        fitness_values = [self.fitness(individual) for individual in self.fitness_values]
        best_individual = max(set(self.fitness_values), key=fitness_values.count)
        return best_individual

class BBOB:
    def __init__(self, func, budget, dim, mutation_rate, selection_rate):
        self.func = func
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.selection_rate = selection_rate
        self.population = Metaheuristic(budget, dim)
        self.selection = Selection(1, dim)

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual
        fitness = self.func(individual)
        self.fitness_values.append(fitness)
        return fitness

    def __call__(self):
        # Run the algorithm for a fixed number of iterations
        for _ in range(100):
            # Select the best individual
            individual = self.selection.select(self.population())

            # Mutate the individual
            individual = Mutation(self.mutation_rate, self.dim).mutate(individual)

            # Evaluate the fitness of the mutated individual
            fitness = self.evaluate_fitness(individual)

            # Update the population
            self.population = Metaheuristic(self.budget, self.dim)

            # Add the fitness value to the population
            self.population.fitness_values.append(fitness)

        # Return the best individual
        return self.population.select(self.population())

# Description: Novel Metaheuristic Algorithm for Black Box Optimization (MMABBO)
# Code: 