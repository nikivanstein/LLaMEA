# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func, budget=1000, max_iter=100):
        population_size = 100
        mutation_rate = 0.01
        crossover_rate = 0.5
        while True:
            fitnesses = [self.evaluate_fitness(individual, func, budget, max_iter) for individual in self.population]
            population = self.select(population_size, fitnesses)
            if np.mean(fitnesses) < -1e-6:  # -inf as score
                break
            new_individuals = self.crossover(population, fitnesses, mutation_rate, crossover_rate)
            population = self.mutate(population, mutation_rate)
            self.population = population
            if len(self.population) > self.budget:
                self.population.pop(0)

    def select(self, population_size, fitnesses):
        # Select individuals based on fitness
        return random.choices(population, weights=fitnesses, k=population_size)

    def crossover(self, population, fitnesses, mutation_rate, crossover_rate):
        # Perform crossover on selected individuals
        new_population = []
        for _ in range(len(population) // 2):
            parent1, parent2 = random.sample(population, 2)
            child1 = parent1[:self.dim]
            child2 = parent2[:self.dim]
            child1[1:] = self.crossover_rate * (parent1[1:] - parent2[1:])
            child2[1:] = self.crossover_rate * (parent2[1:] - parent1[1:])
            new_population.append(child1 + child2)
        return new_population

    def mutate(self, population, mutation_rate):
        # Mutate individuals with probability
        return [individual if random.random() < mutation_rate else random.choice(population) for individual in population]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
bbo_bopt = BBOBOptimizer(1000, 10)