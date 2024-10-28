import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.elite_size = 10
        self.population = self.generate_population()
        self.elite = self.population[:self.elite_size]
        self.fitness_history = []

    def generate_population(self):
        population = []
        for _ in range(self.population_size):
            dim = self.dim
            for _ in range(dim):
                population.append(np.random.uniform(-5.0, 5.0))
        return population

    def fitness_func(self, func, x):
        return evaluate_func(x, func)

    def __call__(self, func, budget):
        def evaluate_func(x, func):
            return func(x)

        while len(self.elite) < self.elite_size:
            # Selection
            fitness_values = [evaluate_func(x, func) for x in self.population]
            indices = np.argsort(fitness_values)[:self.population_size]
            self.elite = [self.population[i] for i in indices]

            # Crossover
            children = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(self.elite, 2)
                child = (parent1 + parent2) / 2
                children.append(child)

            # Mutation
            for child in children:
                if random.random() < 0.1:
                    index = random.randint(0, self.dim - 1)
                    child[index] += random.uniform(-1.0, 1.0)

            # Replace the elite with the children
            self.elite = children

            # Update fitness history
            self.fitness_history.append(evaluate_func(self.elite[0], func))

        # Refine strategy based on fitness history
        best_x = self.elite[0]
        best_fitness = self.fitness_func(best_x, func)
        best_index = np.argmax(self.fitness_history)
        if random.random() < 0.45:
            best_x = self.elite[best_index]
            best_fitness = self.fitness_func(best_x, func)
        return best_x, best_fitness

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 