import numpy as np
import random

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
                individual = np.clip(individual + random.uniform(-1.0, 1.0), bounds[individual].min(), bounds[individual].max())
            return individual

        def crossover(parent1, parent2):
            if random.random() < 0.5:
                child = np.random.choice(len(parent1), size=len(parent1), replace=False)
                child[0] = parent1[child]
                child[1:] = parent2[child]
                return child
            else:
                child = np.concatenate((parent1[:len(parent1)//2], parent2[len(parent2)//2:]))
                return child

        def selection(population):
            return np.array([individual for individual in population if individual not in population[:len(population)//2]])

        def evaluate_fitness(individual, func):
            fitness = objective(individual)
            updated_individual = individual
            for _ in range(self.budget):
                for _ in range(1, len(individual)):
                    new_individual = mutate(updated_individual)
                    fitness = objective(new_individual)
                    if fitness < fitness + 1e-6:
                        updated_individual = new_individual
                new_individual = crossover(updated_individual, individual)
                fitness = objective(new_individual)
                if fitness < fitness + 1e-6:
                    updated_individual = new_individual
            return updated_individual, fitness

        for _ in range(self.budget):
            population = selection(population)
            individual, fitness = evaluate_fitness(population[0], func)
            self.fitnesses[population.index(individual)] = fitness
            self.population = population

        return self.fitnesses

# One-line description: Novel NNEO Algorithm using Genetic Algorithm and Mutation to Refine Individual Strategy