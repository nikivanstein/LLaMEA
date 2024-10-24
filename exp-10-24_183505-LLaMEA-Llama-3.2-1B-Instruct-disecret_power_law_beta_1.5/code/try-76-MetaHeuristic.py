# Description: "Black Box Optimization using Evolutionary Algorithms"
# Code: 
# ```python
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
        # Return the best function found
        return self.best_func

    def set_budget(self, budget):
        self.budget = budget

    def get_best_func(self):
        return self.best_func

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]  # Define the search space
        self.population_size = 100
        self.mutation_rate = 0.01
        self.population = self.generate_population()

    def generate_population(self):
        # Initialize the population with random individuals
        population = []
        for _ in range(self.population_size):
            individual = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            population.append(individual)
        return population

    def evaluate_fitness(self, individual, func):
        # Evaluate the function at the individual
        fitness = func(individual)
        return fitness

    def select_parents(self, fitness):
        # Select parents using tournament selection
        parents = []
        for _ in range(self.population_size):
            individual1 = random.choice(self.population)
            individual2 = random.choice(self.population)
            fitness1 = self.evaluate_fitness(individual1, func)
            fitness2 = self.evaluate_fitness(individual2, func)
            if fitness1 < fitness2:
                parents.append(individual1)
            else:
                parents.append(individual2)
        return parents

    def crossover(self, parents):
        # Perform crossover to create offspring
        offspring = []
        while len(offspring) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = (self.evaluate_fitness(parent1, func) + self.evaluate_fitness(parent2, func)) / 2
            offspring.append(child)
        return offspring

    def mutate(self, offspring):
        # Perform mutation on offspring
        mutated_offspring = []
        for individual in offspring:
            if random.random() < self.mutation_rate:
                mutated_individual = (individual[0] + random.uniform(-1, 1), individual[1] + random.uniform(-1, 1))
                mutated_offspring.append(mutated_individual)
            else:
                mutated_offspring.append(individual)
        return mutated_offspring

    def run(self):
        # Run the genetic algorithm
        for _ in range(1000):
            parents = self.select_parents(self.evaluate_fitness(self.population, func))
            offspring = self.crossover(parents)
            mutated_offspring = self.mutate(offspring)
            self.population = mutated_offspring
        # Return the best individual found
        return self.population[0]

# Define a noiseless function
def noiseless_func(x):
    return np.sin(x)

# Define a noise function
def noise_func(x):
    return np.random.normal(0, 1, x)

# Define a test function
def test_func(x):
    return x**2 + 2*x + 1

# Create an instance of the MetaHeuristic class
meta_heuristic = MetaHeuristic(100, 10)

# Set the budget for the MetaHeuristic
meta_heuristic.set_budget(100)

# Optimize the test function using the MetaHeuristic
best_func = meta_heuristic(__call__, 100)

# Print the best function found
print("Best function:", best_func)
print("Best fitness:", best_func.budget)

# Print the best individual found
print("Best individual:", best_func)
print("Best fitness:", best_func.budget)