# Description: "Black Box Optimization using Evolutionary Strategies"
# Code: 
# ```python
import random
import numpy as np
import matplotlib.pyplot as plt

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

class EvolutionaryStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]  # Define the search space
        self.best_func = None  # Initialize the best function found so far
        self.best_fitness = float('inf')  # Initialize the best fitness found so far
        self.iterations = 0  # Initialize the number of iterations
        self.population = self.generate_population()

    def generate_population(self):
        population = []
        for _ in range(1000):
            individual = self.generate_individual()
            population.append(individual)
        return population

    def generate_individual(self):
        # Create an individual by sampling from the search space
        individual = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
        return individual

    def mutate(self, individual):
        # Mutate an individual by changing a random element
        mutated_individual = individual
        mutated_individual[0] = random.uniform(self.search_space[0], self.search_space[1])
        mutated_individual[1] = random.uniform(self.search_space[0], self.search_space[1])
        return mutated_individual

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of an individual
        fitness = self.func(individual)
        return fitness

    def select_parents(self, population):
        # Select parents for the next generation
        parents = []
        for _ in range(int(self.budget * 0.1)):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            while parent1 == parent2:
                parent2 = random.choice(population)
            parents.append((parent1, parent2))
        return parents

    def crossover(self, parents):
        # Perform crossover between parents
        offspring = []
        for _ in range(int(self.budget * 0.7)):
            parent1, parent2 = random.sample(parents, 2)
            child = (parent1[0] + parent2[0]) / 2, (parent1[1] + parent2[1]) / 2
            offspring.append(child)
        return offspring

    def mutate_offspring(self, offspring):
        # Mutate offspring
        mutated_offspring = []
        for individual in offspring:
            mutated_individual = self.mutate(individual)
            mutated_offspring.append(mutated_individual)
        return mutated_offspring

    def evolve(self):
        # Evolve the population
        population = self.population
        for _ in range(int(self.budget * 0.9)):
            parents = self.select_parents(population)
            offspring = self.crossover(parents)
            mutated_offspring = self.mutate_offspring(offspring)
            population = self.evaluate_fitness(mutated_offspring)
        return population

# Description: "Black Box Optimization using Evolutionary Strategies"
# Code: 
# ```python
# import random
# import numpy as np
# import matplotlib.pyplot as plt

# Define the MetaHeuristic class
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

# Evolve the population
evolved_population = meta_heuristic.evolve()

# Print the best function found in the evolved population
print("Best function in evolved population:", evolved_population[0])
print("Best fitness in evolved population:", evolved_population[0].budget)