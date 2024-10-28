# AdaptiveBlackBoxOptimizer: Adaptive Black Box Optimization using Evolutionary Strategies
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
        # Use a simple strategy to generate initial population
        # Replace this with a more sophisticated strategy
        return np.random.uniform(self.search_space, self.search_space, (self.population_size, self.dim))

    def fitness_function(self, func, x):
        # Evaluate the fitness of a function at a given point
        return func(x)

    def __call__(self, func):
        def objective(x):
            return self.fitness_function(func, x)

        def fitness(x):
            return objective(x)

        def selection(self, fitness_values):
            return np.random.choice(self.population_size, self.population_size, p=fitness_values)

        def crossover(self, parent1, parent2):
            child = np.zeros(self.dim)
            for i in range(self.dim):
                child[i] = (parent1[i] + parent2[i]) / 2
            return child

        def mutation(self, child):
            for i in range(self.dim):
                if random.random() < 0.1:
                    child[i] += random.uniform(-1, 1)
            return child

        def selection_with_crossover(self, fitness_values, parent1, parent2):
            selected_indices = selection(fitness_values)
            child1 = parent1[selected_indices]
            child2 = parent2[selected_indices]
            child = crossover(child1, child2)
            child = mutation(child)
            return child

        def selection_with_crossover_and_mutation(self, fitness_values, parent1, parent2):
            selected_indices = selection(fitness_values)
            child1 = parent1[selected_indices]
            child2 = parent2[selected_indices]
            child = crossover(child1, child2)
            child = mutation(child)
            return child

        def evolution(self, population, fitness_values, num_generations):
            # Use a simple strategy to evolve the population
            # Replace this with a more sophisticated strategy
            for _ in range(num_generations):
                population = self.select(population, fitness_values)
                population = self.crossover(population)
                population = self.mutation(population)
            return population

        def select(self, fitness_values, num_indices):
            fitness_values = fitness_values / np.max(fitness_values)
            selected_indices = np.random.choice(self.population_size, num_indices, p=fitness_values)
            return self.population[selected_indices]

        def crossover(self, parent1, parent2):
            child = np.zeros(self.dim)
            for i in range(self.dim):
                child[i] = (parent1[i] + parent2[i]) / 2
            return child

        def mutation(self, child):
            for i in range(self.dim):
                if random.random() < 0.1:
                    child[i] += random.uniform(-1, 1)
            return child

        self.population = self.evolve(self.population, fitness_values, num_generations)

    def evolve(self, population, fitness_values, num_generations):
        population = self.select(population, fitness_values)
        for _ in range(num_generations):
            population = self.crossover(population)
            population = self.mutation(population)
        return population

    def run(self, func):
        while True:
            population = self.evolve(self.fitness_values, self.fitness_function, 100)
            if all(self.fitness_values >= 0.5):
                break
        return population

def _select_with_crossover_and_mutation(func, population, fitness_values, num_indices, parent1, parent2):
    selected_indices = selection_with_crossover(fitness_values, parent1, parent2)
    child1 = parent1[selected_indices]
    child2 = parent2[selected_indices]
    child = crossover(child1, child2)
    child = mutation(child)
    return child

def _select_with_crossover(func, population, fitness_values, num_indices):
    selected_indices = selection(fitness_values)
    child1 = population[selected_indices]
    child2 = population[selected_indices]
    child = crossover(child1, child2)
    child = mutation(child)
    return child

def _evolve(func, population, fitness_values, num_generations):
    for _ in range(num_generations):
        population = _select_with_crossover_and_mutation(func, population, fitness_values, 50, population, population)
    return population

# Test the new algorithm
def test_func(x):
    return x**2

optimizer = AdaptiveBlackBoxOptimizer(100, 5)
optimizer.run(test_func)