import random
import numpy as np

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()
        self.population_size = 100
        self.population = self.initialize_population()
        self.selector = self.adaptive_selector()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def initialize_population(self):
        return [self.funcs[np.random.choice(len(self.funcs))] for _ in range(self.population_size)]

    def adaptive_selector(self):
        if random.random() < 0.4:
            return random.choice(self.population)
        else:
            # Adaptive selection: choose the best individual based on the best fitness in the last 20% of the population
            last_20_percent = self.population[-20:]
            best_individual = max(last_20_percent, key=lambda x: self.evaluate_fitness(x))
            return best_individual

    def evaluate_fitness(self, individual):
        return self.func(individual)

    def mutate(self, individual):
        # Randomly mutate the individual by changing a random element
        return individual[:random.randint(0, len(individual)-1)] + [random.uniform(-5.0, 5.0) for _ in range(len(individual))]

    def __call__(self, func):
        # Optimize the black box function using the adaptive selector and ensemble search
        for _ in range(self.budget):
            new_individual = self.selector(func)
            new_fitness = self.evaluate_fitness(new_individual)
            if new_fitness < func(new_individual):
                new_individual = new_individual
            self.population.append(new_individual)
            self.population.sort(key=self.evaluate_fitness, reverse=True)
            self.population.pop(0)
        return self.population[0]

# Description: Black Box Optimization using BBOB
# Code: 