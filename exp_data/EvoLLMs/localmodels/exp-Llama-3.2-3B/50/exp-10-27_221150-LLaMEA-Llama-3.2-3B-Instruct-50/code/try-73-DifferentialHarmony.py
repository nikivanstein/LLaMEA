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
        self.population = self.initialize_population()
        self.probability_adjustment = 0.5

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
                new_x = self.harmony_search(self.population[i], x1, x2, x3, x4)
                if random.random() < self.probability_adjustment:
                    new_x = self.differential_evolution(x1, x2, x3, x4, new_x)
                self.population[i] = new_x

            # Harmony Search
            best_x = self.population[np.argmin([func(x) for x in self.population])]
            self.population = self.update_population(self.population, best_x)

    def initialize_population(self):
        return [np.random.uniform(self.lower_bound, self.upper_bound, self.dim) for _ in range(self.population_size)]

    def harmony_search(self, x, x1, x2, x3, x4):
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

    def differential_evolution(self, x1, x2, x3, x4, new_x):
        # Calculate the differential evolution parameters
        mu1 = random.random()
        mu2 = random.random()
        sigma1 = random.random()
        sigma2 = random.random()

        # Calculate the new individual using differential evolution
        new_x1 = (mu1 * new_x + sigma1 * x1) % self.dim
        new_x2 = (mu2 * new_x + sigma2 * x2) % self.dim
        new_x3 = (mu1 * new_x + sigma1 * x3) % self.dim
        new_x4 = (mu2 * new_x + sigma2 * x4) % self.dim

        # Return the new individual
        return new_x1, new_x2, new_x3, new_x4

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
                new_x = self.harmony_search(x1, x2, x3, x4, best_x)
                if random.random() < self.probability_adjustment:
                    new_x = self.differential_evolution(x1, x2, x3, x4, new_x)
                new_population.append(new_x)
            else:
                new_population.append(best_x)
        return new_population

# Example usage
def func(x):
    return np.sum(x**2)

differential_harmony = DifferentialHarmony(100, 10)
differential_harmony(func)