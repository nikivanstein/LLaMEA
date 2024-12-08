import numpy as np
import random

class MEEAM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.stochastic_search_rate = 0.1
        self.fitness_values = []
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Select the best points for the next generation
            next_generation = self.fitness_values[:int(self.population_size * 0.2)]

            # Perform crossover and mutation
            for i in range(self.population_size):
                if random.random() < self.crossover_rate:
                    parent1, parent2 = random.sample(next_generation, 2)
                    child = np.mean([parent1[0], parent2[0]], axis=0)
                    if random.random() < self.mutation_rate:
                        child += np.random.uniform(-0.1, 0.1, self.dim)
                else:
                    child = next_generation[i][0]

                # Evaluate the fitness of the child using stochastic search
                if random.random() < self.stochastic_search_rate:
                    child = self.stochastic_search(func, child, self.population)

                # Evaluate the fitness of the child
                fitness = func(child)
                self.fitness_values.append((child, fitness))

            # Sort the population based on fitness
            self.fitness_values.sort(key=lambda x: x[1])

            # Update the population
            self.population = np.array([x[0] for x in self.fitness_values])

        # Return the best point in the population
        return self.fitness_values[-1][0]

    def stochastic_search(self, func, point, population):
        # Generate a new point by perturbing the current point
        new_point = point + np.random.uniform(-0.1, 0.1, self.dim)

        # Evaluate the fitness of the new point
        fitness = func(new_point)

        # If the new point has a better fitness, replace the current point
        if fitness < func(point):
            return new_point
        else:
            return point

# Example usage:
def func(x):
    return np.sum(x**2)

meeam = MEEAM(budget=100, dim=10)
best_point = meeam(func)
print(best_point)