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
        # Create a population of random individuals with search space [-5.0, 5.0]
        return np.random.uniform(self.search_space, self.search_space, (self.population_size, self.dim))

    def fitness_function(self, func, x):
        # Evaluate the fitness of an individual using the given function
        return func(x)

    def __call__(self, func):
        # Define the objective function to optimize
        def objective(x):
            return self.fitness_function(func, x)

        def fitness(x):
            return objective(x)

        def selection(self, fitness_values):
            # Select the fittest individuals using tournament selection
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
            # Select the fittest individuals using tournament selection and crossover
            selected_indices = selection(fitness_values)
            child1 = parent1[selected_indices]
            child2 = parent2[selected_indices]
            child = crossover(child1, child2)
            child = mutation(child)
            return child

        def selection_with_crossover_and_mutation(self, fitness_values, parent1, parent2):
            # Select the fittest individuals using tournament selection, crossover, and mutation
            selected_indices = selection(fitness_values)
            child1 = parent1[selected_indices]
            child2 = parent2[selected_indices]
            child = crossover(child1, child2)
            child = mutation(child)
            return child

        def evolution(self, population, fitness_values, num_generations):
            # Evolve the population using tournament selection, crossover, and mutation
            for _ in range(num_generations):
                population = self.select(population, fitness_values)
                population = self.crossover(population)
                population = self.mutation(population)
            return population

        def select(self, fitness_values, num_indices):
            # Select the fittest individuals using tournament selection
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

        # Evolve the population
        self.population = self.evolve(self.fitness_values, self.fitness_function, 100)

    def evolve(self, population, fitness_values, num_generations):
        # Evolve the population using tournament selection, crossover, and mutation
        for _ in range(num_generations):
            population = self.select(population, fitness_values)
            population = self.crossover(population)
            population = self.mutation(population)
        return population

    def run(self, func):
        # Run the optimization algorithm on the given function
        while True:
            population = self.evolve(self.fitness_values, self.fitness_function, 100)
            if all(self.fitness_values >= 0.5):
                break
        return population

def black_box_optimization(func, budget, dim):
    # Define the optimization algorithm using the given function
    return AdaptiveBlackBoxOptimizer(budget, dim)

# Example usage:
def example_function(x):
    return x**2

black_box_optim = black_box_optimization(example_function, 1000, 10)
population = black_box_optim.run(example_function)
print(population)