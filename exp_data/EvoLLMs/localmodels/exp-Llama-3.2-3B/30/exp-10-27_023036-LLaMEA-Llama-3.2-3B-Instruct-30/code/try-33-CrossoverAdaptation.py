import numpy as np
import random

class CrossoverAdaptation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.fitness_values = [func(x) for x in self.population[0]]
        self.best_solution = self.population[0]
        self.iteration = 0

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def evaluate(self, func):
        self.fitness_values = [func(x) for x in self.population]
        self.best_solution = self.population[np.argmin(self.fitness_values)]
        self.iteration += 1

    def crossover(self, parent1, parent2):
        child = parent1 + parent2
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent1[i]
        return child

    def mutation(self, individual):
        mutation_rate = 0.1
        for i in range(self.dim):
            if random.random() < mutation_rate:
                individual[i] = np.random.uniform(-5.0, 5.0)
        return individual

    def adapt(self):
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(self.population, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutation(child)
            new_population.append(child)
        self.population = new_population

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
            self.adapt()
            if self.iteration % 10 == 0:
                self.fitness_values.sort()
                print(f"Iteration {_}, Best fitness: {self.fitness_values[0]}")
        return self.best_solution

# Example usage
if __name__ == "__main__":
    func = lambda x: np.sum(x**2)
    ca = CrossoverAdaptation(budget=100, dim=10)
    solution = ca(func)
    print("Optimal solution:", solution)