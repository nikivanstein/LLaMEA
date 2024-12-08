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

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        return self.fitnesses

    def mutate(self, individual):
        if random.random() < 0.2:
            mutation = random.uniform(-5.0, 5.0)
            individual[0] += mutation
            individual[1] += mutation
            if individual[0] < -5.0:
                individual[0] = -5.0
            elif individual[0] > 5.0:
                individual[0] = 5.0
            if individual[1] < -5.0:
                individual[1] = -5.0
            elif individual[1] > 5.0:
                individual[1] = 5.0
        return individual

    def evaluate_fitness(self, individual):
        updated_individual = self.evaluate_individual(individual)
        fitness = objective(updated_individual)
        return fitness, updated_individual

    def evaluate_individual(self, individual):
        updated_individual = individual
        for i in range(self.dim):
            if random.random() < 0.2:
                mutation = random.uniform(-5.0, 5.0)
                updated_individual[i] += mutation
                if updated_individual[i] < -5.0:
                    updated_individual[i] = -5.0
                elif updated_individual[i] > 5.0:
                    updated_individual[i] = 5.0
        return updated_individual

    def select_solution(self, individual):
        selected_individual = individual
        for i in range(self.dim):
            if random.random() < 0.2:
                mutation = random.uniform(-5.0, 5.0)
                selected_individual[i] += mutation
                if selected_individual[i] < -5.0:
                    selected_individual[i] = -5.0
                elif selected_individual[i] > 5.0:
                    selected_individual[i] = 5.0
        return selected_individual

    def update(self, func):
        while True:
            individual = random.choice(self.population)
            fitness, updated_individual = self.evaluate_fitness(individual)
            selected_individual = self.select_solution(updated_individual)
            new_individual = self.mutate(selected_individual)
            self.population.append(new_individual)
            self.fitnesses.append(fitness)
            if self.fitnesses[-1] > fitness:
                break

# Example usage
func = lambda x: x**2
nneo = NNEO(100, 2)
nneo.update(func)
print(nneo.fitnesses[-1])
print(nneo.population)