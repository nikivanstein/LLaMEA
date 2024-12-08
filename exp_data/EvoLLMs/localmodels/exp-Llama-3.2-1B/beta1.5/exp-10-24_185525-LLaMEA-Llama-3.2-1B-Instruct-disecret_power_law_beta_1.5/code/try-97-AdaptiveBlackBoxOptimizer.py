# Description: Adaptive Black Box Optimization using Evolutionary Strategies
# Code: 
# ```python
import random
import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.generate_initial_population()
        self.fitness_values = np.zeros((self.population_size, self.dim))
        self.search_space = np.linspace(-5.0, 5.0, self.dim)

    def generate_initial_population(self):
        # Use the given search space for the initial population
        return np.random.uniform(self.search_space, self.search_space, (self.population_size, self.dim))

    def fitness_function(self, func, x):
        # Evaluate the fitness of a function at a given point
        return func(x)

    def __call__(self, func):
        # Define the objective function to be optimized
        def objective(x):
            return self.fitness_function(func, x)

        def fitness(x):
            return objective(x)

        def selection(self, fitness_values):
            # Select the fittest individuals
            return np.random.choice(self.population_size, self.population_size, p=fitness_values)

        def crossover(self, parent1, parent2):
            # Perform crossover between two parents
            child = np.zeros(self.dim)
            for i in range(self.dim):
                child[i] = (parent1[i] + parent2[i]) / 2
            return child

        def mutation(self, child):
            # Perform mutation on a child
            for i in range(self.dim):
                if random.random() < 0.1:
                    child[i] += random.uniform(-1, 1)
            return child

        def selection_with_crossover(self, fitness_values, parent1, parent2):
            # Select the fittest individuals using crossover
            selected_indices = selection(fitness_values)
            child1 = parent1[selected_indices]
            child2 = parent2[selected_indices]
            child = crossover(child1, child2)
            child = mutation(child)
            return child

        def selection_with_crossover_and_mutation(self, fitness_values, parent1, parent2):
            # Select the fittest individuals using crossover and mutation
            selected_indices = selection(fitness_values)
            child1 = parent1[selected_indices]
            child2 = parent2[selected_indices]
            child = crossover(child1, child2)
            child = mutation(child)
            return child

        def evolution(self, population, fitness_values, num_generations):
            # Perform evolution for a specified number of generations
            for _ in range(num_generations):
                population = self.select(population, fitness_values)
                population = self.crossover(population)
                population = self.mutation(population)
            return population

        def select(self, fitness_values, num_indices):
            # Select the fittest individuals using fitness proportion
            fitness_values = fitness_values / np.max(fitness_values)
            selected_indices = np.random.choice(self.population_size, num_indices, p=fitness_values)
            return self.population[selected_indices]

        def crossover(self, parent1, parent2):
            # Perform crossover between two parents
            child = np.zeros(self.dim)
            for i in range(self.dim):
                child[i] = (parent1[i] + parent2[i]) / 2
            return child

        def mutation(self, child):
            # Perform mutation on a child
            for i in range(self.dim):
                if random.random() < 0.1:
                    child[i] += random.uniform(-1, 1)
            return child

        # Select the fittest individuals using fitness proportion
        self.population = self.evolve(self.fitness_values, self.fitness_function, 100)
        # Refine the strategy to change the individual lines of the selected solution
        self.population = self.evolve(self.population, self.fitness_function, 100)
        # Refine the strategy to change the individual lines of the selected solution
        self.population = self.evolve(self.population, self.fitness_function, 100)

    def evolve(self, population, fitness_function, num_generations):
        # Perform evolution for a specified number of generations
        for _ in range(num_generations):
            population = self.select(population, fitness_function, self.population_size)
            population = self.crossover(population)
            population = self.mutation(population)
        return population

    def run(self, func):
        # Run the optimization algorithm for a specified number of function evaluations
        while True:
            population = self.evolve(self.fitness_function, func, 100)
            if all(self.fitness_function(population, func) >= 0.5):
                break
        return population

# Description: Adaptive Black Box Optimization using Evolutionary Strategies
# Code: 
# ```python
def objective(x):
    return np.sum(x**2)

def fitness_function(func, x):
    return np.sum(func(x)**2)

def selection_with_crossover(fitness_values, parent1, parent2):
    selected_indices = np.random.choice(fitness_values.shape[0], fitness_values.shape[0], p=fitness_values)
    child1 = parent1[selected_indices]
    child2 = parent2[selected_indices]
    child = np.concatenate((child1, child2))
    return child

def mutation(child):
    for i in range(child.shape[0]):
        if random.random() < 0.1:
            child[i] += random.uniform(-1, 1)
    return child

# Create an instance of the AdaptiveBlackBoxOptimizer class
optimizer = AdaptiveBlackBoxOptimizer(100, 10)

# Run the optimization algorithm
optimizer.run(objective)