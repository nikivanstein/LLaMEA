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
        self.evolution = False

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(x):
            if random.random() < 0.2:
                return np.clip(x + np.random.uniform(-1.0, 1.0), bounds[x].min(), bounds[x].max())
            else:
                return x

        def crossover(parent1, parent2):
            if random.random() < 0.5:
                return np.concatenate((parent1[:random.randint(0, self.dim)], parent2[random.randint(0, self.dim)]))
            else:
                return np.concatenate((parent1, parent2[:random.randint(0, self.dim)]))

        def selection(population):
            return np.random.choice(len(population), size=self.population_size, replace=False)

        def evaluate_fitness(individual, problem):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual] + 1e-6:
                self.fitnesses[individual] = fitness
                return individual
            else:
                return individual

        if not self.evolution:
            while True:
                for _ in range(self.budget):
                    for i in range(self.population_size):
                        if not self.evolution:
                            new_individual = copy.deepcopy(self.population[i])
                        else:
                            new_individual = random.choices(self.population, k=1)[0]
                        new_individual = mutate(new_individual)
                        new_individual = crossover(new_individual, new_individual)
                        new_individual = selection([individual for individual in self.population if individual!= new_individual])
                        new_individual = evaluate_fitness(new_individual, problem)
                        if new_individual!= new_individual:
                            self.population[i] = new_individual
                            self.fitnesses[i] = self.fitnesses[new_individual]
                self.evolution = True
                break

        new_individual = evaluate_fitness(copy.deepcopy(self.population[0]), func)
        new_individual = mutate(new_individual)
        new_individual = crossover(new_individual, new_individual)
        new_individual = selection([individual for individual in self.population if individual!= new_individual])
        new_individual = evaluate_fitness(new_individual, func)
        if new_individual!= new_individual:
            self.population[0] = new_individual
            self.fitnesses[0] = self.fitnesses[new_individual]
        return new_individual