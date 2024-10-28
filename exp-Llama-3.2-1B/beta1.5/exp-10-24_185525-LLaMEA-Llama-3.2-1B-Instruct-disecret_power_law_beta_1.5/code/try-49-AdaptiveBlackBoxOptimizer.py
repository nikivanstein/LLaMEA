# Description: Adaptive Black Box Optimization using Evolutionary Strategies with Adaptive Mutation
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
        return np.random.uniform(self.search_space, self.search_space, (self.population_size, self.dim))

    def fitness_function(self, func, x):
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

        self.population = self.evolve(self.fitness_values, self.fitness_function, 100)

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

def fitness_function(bbo, func, x):
    return bbo.run(func)

class AdaptiveBlackBoxOptimizerWithAdaptiveMutation(AdaptiveBlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = 0.2

    def __call__(self, func):
        def objective(x):
            return fitness_function(self, func, x)

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
                if random.random() < self.mutation_rate:
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
                if random.random() < 0.2:
                    child[i] += random.uniform(-1, 1)
            return child

        self.population = self.evolve(self.fitness_values, self.fitness_function, 100)

        # Refine the strategy with adaptive mutation
        def select_with_adaptive_mutation(self, fitness_values, num_indices):
            fitness_values = fitness_values / np.max(fitness_values)
            selected_indices = np.random.choice(self.population_size, num_indices, p=fitness_values)
            mutation_rate = 0.2
            for _ in range(int(num_indices * mutation_rate)):
                child = np.zeros(self.dim)
                for i in range(self.dim):
                    if random.random() < mutation_rate:
                        child[i] += random.uniform(-1, 1)
                selected_indices = np.random.choice(self.population_size, num_indices, p=fitness_values)
                child = self.population[selected_indices]
            return self.population[selected_indices]

        self.population = select_with_adaptive_mutation(self.population, self.population_size)

def fitness_function(bbo, func, x):
    return bbo.run(func)

class AdaptiveBlackBoxOptimizerWithAdaptiveMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.generate_initial_population()
        self.fitness_values = np.zeros((self.population_size, self.dim))
        self.search_space = np.linspace(-5.0, 5.0, self.dim)

    def generate_initial_population(self):
        return np.random.uniform(self.search_space, self.search_space, (self.population_size, self.dim))

    def fitness_function(self, func, x):
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

        self.population = self.evolve(self.fitness_values, self.fitness_function, 100)

        def select_with_crossover(self, fitness_values, parent1, parent2):
            selected_indices = selection(fitness_values)
            child1 = parent1[selected_indices]
            child2 = parent2[selected_indices]
            child = crossover(child1, child2)
            child = mutation(child)
            return child

        self.population = select_with_crossover(self.population, self.population_size)

        def select_with_adaptive_mutation(self, fitness_values, num_indices):
            fitness_values = fitness_values / np.max(fitness_values)
            selected_indices = np.random.choice(self.population_size, num_indices, p=fitness_values)
            mutation_rate = 0.2
            for _ in range(int(num_indices * mutation_rate)):
                child = np.zeros(self.dim)
                for i in range(self.dim):
                    if random.random() < mutation_rate:
                        child[i] += random.uniform(-1, 1)
                selected_indices = np.random.choice(self.population_size, num_indices, p=fitness_values)
                child = self.population[selected_indices]
            return self.population[selected_indices]

        self.population = select_with_adaptive_mutation(self.population, self.population_size)

# Run the adaptive mutation algorithm
adaptive_bbo = AdaptiveBlackBoxOptimizerWithAdaptiveMutation(budget=1000, dim=10)
adaptive_bbo.run(lambda func: func(np.random.uniform(-10, 10, 10)))