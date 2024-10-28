import numpy as np
import random

class DABU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.learning_rate = 0.01
        self.diversity = 0.1
        self.population_size = 100

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def evolve(self, population):
        # Select parents using tournament selection
        parents = random.sample(population, self.population_size)
        for _ in range(10):  # evolve for 10 generations
            # Create offspring by crossover and mutation
            offspring = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                child = (0.5 * (parent1 + parent2)) / 2
                if random.random() < self.diversity:
                    child[0] += random.uniform(-1, 1)
                    child[1] += random.uniform(-1, 1)
                offspring.append(child)
            population = offspring

        # Replace least fit individuals with new ones
        population = [func(self.search_space) for func in population]

        # Evolve for another generation
        self.learning_rate *= 0.9
        self.diversity *= 0.9

        return population

    def optimize(self, func):
        population = [func(self.search_space) for func in self.evolve([func])]
        return self.__call__(func)