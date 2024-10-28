import numpy as np
import random

class DifferentialHarmony:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.differential_evolution_rate = 0.5
        self.harmony_search_rate = 0.2
        self.probability = 0.5
        self.population = self.initialize_population()

    def __call__(self, func):
        for _ in range(self.budget):
            if len(self.population) < self.population_size:
                self.population = self.initialize_population()

            # Differential Evolution
            for i in range(self.population_size):
                x1, x2 = random.sample(range(self.dim), 2)
                x3, x4 = random.sample(range(self.dim), 2)
                while x1 == x3 or x1 == x2 or x3 == x4 or x3 == x2:
                    x1, x2 = random.sample(range(self.dim), 2)
                    x3, x4 = random.sample(range(self.dim), 2)
                x1, x2, x3, x4 = self.population[x1], self.population[x2], self.population[x3], self.population[x4]
                self.population[i] = self.harmony_search(self.population[i], x1, x2, x3, x4, func)

            # Harmony Search
            best_x = self.population[np.argmin([func(x) for x in self.population])]
            self.population = self.update_population(self.population, best_x)

    def initialize_population(self):
        return [np.random.uniform(self.lower_bound, self.upper_bound, self.dim) for _ in range(self.population_size)]

    def harmony_search(self, x, x1, x2, x3, x4, func):
        r1 = random.random()
        r2 = random.random()
        if r1 < self.harmony_search_rate:
            return x1
        elif r2 < self.harmony_search_rate:
            return x2
        elif r1 < self.harmony_search_rate + self.harmony_search_rate:
            return x3
        else:
            return x4
        if random.random() < self.probability:
            return self.differential_evolution(x, x1, x2, x3, x4, func)

    def differential_evolution(self, x, x1, x2, x3, x4, func):
        # Calculate the fitness of the current individual
        f_x = func(x)
        # Calculate the fitness of the neighboring individuals
        f_x1 = func(x1)
        f_x2 = func(x2)
        f_x3 = func(x3)
        f_x4 = func(x4)
        # Calculate the difference between the fitness of the current individual and the neighboring individuals
        dx1 = f_x - f_x1
        dx2 = f_x - f_x2
        dx3 = f_x - f_x3
        dx4 = f_x - f_x4
        # Calculate the relative difference between the fitness of the neighboring individuals
        rx1 = dx1 / (f_x1 + 1e-6)
        rx2 = dx2 / (f_x2 + 1e-6)
        rx3 = dx3 / (f_x3 + 1e-6)
        rx4 = dx4 / (f_x4 + 1e-6)
        # Calculate the new individual using the differential evolution strategy
        new_x = x + (rx1 * (x1 - x) + rx2 * (x2 - x) + rx3 * (x3 - x) + rx4 * (x4 - x)) * random.random()
        return new_x

    def update_population(self, population, best_x):
        new_population = []
        for i in range(self.population_size):
            x = random.random() < self.differential_evolution_rate
            if x:
                x1, x2 = random.sample(range(self.dim), 2)
                x3, x4 = random.sample(range(self.dim), 2)
                while x1 == x3 or x1 == x2 or x3 == x4 or x3 == x2:
                    x1, x2 = random.sample(range(self.dim), 2)
                    x3, x4 = random.sample(range(self.dim), 2)
                x1, x2, x3, x4 = population[x1], population[x2], population[x3], population[x4]
                new_population.append(self.harmony_search(x, x1, x2, x3, x4, func))
            else:
                new_population.append(best_x)
        return new_population

# Example usage
def func(x):
    return np.sum(x**2)

differential_harmony = DifferentialHarmony(100, 10)
differential_harmony(func)