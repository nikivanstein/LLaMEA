# Description: Adaptive Black Box Optimization using Evolutionary Strategies with Refinement
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
        if self.dim == 2:
            return np.random.uniform(self.search_space, self.search_space, (self.population_size, 2))
        elif self.dim == 3:
            return np.random.uniform(self.search_space, self.search_space, (self.population_size, 3))
        else:
            raise ValueError("Invalid dimensionality")

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

        def refine_selection(self, selection_values):
            best_individual = np.argmax(selection_values)
            worst_individual = np.argmin(selection_values)
            if self.dim == 2:
                self.population[best_individual] = np.mean(self.population[best_individual], axis=0)
                self.population[worst_individual] = np.mean(self.population[worst_individual], axis=0)
            elif self.dim == 3:
                self.population[best_individual] = np.mean(self.population[best_individual], axis=0)
                self.population[worst_individual] = np.mean(self.population[worst_individual], axis=0)
            else:
                raise ValueError("Invalid dimensionality")

        def run_with_refinement(self, func):
            while True:
                population = self.evolve(self.fitness_values, self.fitness_function, 100)
                if all(self.fitness_values >= 0.5):
                    break
            return population

    def run(self, func):
        while True:
            population = self.run_with_refinement(func)
            if all(self.fitness_values >= 0.5):
                break
        return population

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = AdaptiveBlackBoxOptimizer(100, 2)
optimizer.run(func)