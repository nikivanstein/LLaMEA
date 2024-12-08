# Black Box Optimization using Evolutionary Algorithm
# Description: A novel metaheuristic algorithm that uses evolutionary strategies to optimize black box functions
# Code: 
# ```python
import numpy as np
import random
import copy

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            if random.random() < 0.2:
                new_individual = copy.deepcopy(individual)
                new_individual[random.randint(0, self.dim-1)] += random.uniform(-1.0, 1.0)
                return new_individual
            else:
                return individual

        def crossover(parent1, parent2):
            if random.random() < 0.5:
                child = np.zeros(self.dim)
                for i in range(self.dim):
                    if random.random() < 0.5:
                        child[i] = parent1[i]
                    else:
                        child[i] = parent2[i]
                return child
            else:
                child = parent1
                return child

        def selection(population):
            return np.array([np.random.choice(len(population), p=[0.2, 0.8]) for _ in range(self.population_size)])

        def mutate_and_crossover(population):
            new_population = []
            for _ in range(self.budget):
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                child = crossover(parent1, parent2)
                new_population.append(mutate(child))
            return new_population

        def fitness(individual):
            x = individual
            fitness = objective(x)
            if fitness < self.fitnesses[individual, x] + 1e-6:
                self.fitnesses[individual, x] = fitness
            return fitness

        for _ in range(self.budget):
            population = selection(population)
            population = mutate_and_crossover(population)

        return fitness(population)

# Test the algorithm
def test_nneo():
    func = lambda x: x**2
    nneo = NNEO(100, 10)
    print("NNEO score:", nneo(neoOBJ(func)))

test_nneo()