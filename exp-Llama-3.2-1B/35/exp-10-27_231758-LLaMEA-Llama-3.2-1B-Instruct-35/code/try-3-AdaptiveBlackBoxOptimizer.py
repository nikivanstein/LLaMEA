import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.search_space = np.linspace(-5.0, 5.0, self.dim)
        self.population_size = 100
        self.population = self.init_population()

    def init_population(self):
        return [np.random.uniform(self.search_space) for _ in range(self.population_size)]

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

        # Refine the strategy based on the current population
        if self.func_evals > 0:
            idx1, idx2 = random.sample(range(self.population_size), 2)
            if self.func_evals > 10:
                self.func_values[idx1] = func(self.func_values[idx1])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break
                self.func_values[idx2] = func(self.func_values[idx2])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break
            else:
                self.func_values[idx1] = func(self.func_values[idx1])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break
                self.func_values[idx2] = func(self.func_values[idx2])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

        # Evolve the population
        self.population = self.evolve_population(self.population)

        # Return the best individual
        return self.population[np.argmax(self.func_values)]

    def evolve_population(self, population):
        # Select the fittest individuals
        fittest = population[np.argsort(self.func_values)]
        
        # Create offspring
        offspring = []
        while len(offspring) < self.population_size:
            parent1, parent2 = random.sample(fittest, 2)
            child = (parent1 + parent2) / 2
            offspring.append(child)
        
        # Mutate the offspring
        for i in range(len(offspring)):
            if random.random() < 0.1:
                offspring[i] += np.random.uniform(-1, 1)
        
        # Replace the least fit individuals with the offspring
        population = fittest[:-len(offspring)] + offspring
        
        return population

    def __str__(self):
        return "AdaptiveBlackBoxOptimizer: (Population: {}\nScore: {}\nFunction: {}\nPopulation Size: {}\nDim: {}\nSearch Space: {}".format(
            self.population,
            self.func_evals,
            self.func_values,
            self.population_size,
            self.dim,
            self.search_space)